import socket
import socketserver
import threading

import pytest

from leadgen.verify.smtp_check import SmtpProber, classify_rcpt


@pytest.mark.parametrize(
    "code,msg,expected",
    [
        (250, b"2.1.5 OK", "accepted"),
        (251, b"User not local; will forward", "accepted"),
        (550, b"5.1.1 <x@acme.com>: Recipient address rejected: User unknown in virtual mailbox table", "rejected"),
        (550, b"5.1.1 The email account that you tried to reach does not exist.", "rejected"),
        (550, b"5.4.1 Recipient address rejected: Access denied. AS(201806281)", "rejected"),
        (550, b"5.2.1 The email account that you tried to reach is disabled.", "rejected"),
        (552, b"5.2.2 Mailbox full", "full"),
        (550, b"5.2.2 The email account that you tried to reach is over quota", "full"),
        (554, b"5.7.1 Service unavailable; Client host [1.2.3.4] blocked using zen.spamhaus.org", "blocked"),
        (550, b"Your IP has a poor reputation", "blocked"),
        (450, b"4.2.0 Greylisted, please try again later", "temp"),
        (421, b"Too many connections", "temp"),
        (252, b"Cannot VRFY user", "temp"),
        (550, b"No such user here", "rejected"),
        (550, b"Rejected", "rejected"),
        (554, b"Transaction failed", "blocked"),
    ],
)
def test_classify_rcpt(code, msg, expected):
    assert classify_rcpt(code, msg) == expected


class FakeSMTP:
    """Scripted stand-in for smtplib.SMTP."""

    def __init__(self, mailboxes, catch_all=False, mail_code=250, rcpt_override=None, fail_connect=False, **_):
        self.mailboxes = mailboxes
        self.catch_all = catch_all
        self.mail_code = mail_code
        self.rcpt_override = rcpt_override or {}
        self.fail_connect = fail_connect
        self.rcpts = []
        self.data_sent = False

    def __call__(self, timeout=None):
        return self

    def connect(self, host, port):
        if self.fail_connect:
            raise ConnectionRefusedError("port 25 blocked")
        return 220, b"mx ready"

    def ehlo(self, name):
        return 250, b"hello"

    def helo(self, name):
        return 250, b"hello"

    def mail(self, sender):
        return self.mail_code, b"ok" if self.mail_code == 250 else b"5.7.1 sender rejected by policy"

    def rcpt(self, addr):
        self.rcpts.append(addr)
        if addr in self.rcpt_override:
            return self.rcpt_override[addr]
        if addr in self.mailboxes or self.catch_all:
            return 250, b"2.1.5 OK"
        return 550, b"5.1.1 user unknown"

    def data(self, *_):  # must never be called
        self.data_sent = True

    def quit(self):
        return 221, b"bye"

    def close(self):
        pass


def _prober(fake):
    return SmtpProber("mail.agency.test", "verify@agency.test", smtp_factory=fake)


def test_probe_confirms_mailbox_and_detects_non_catch_all():
    fake = FakeSMTP({"jane@acme.com"})
    probe = _prober(fake).probe("acme.com", ["mx1.acme.com"], ["jane@acme.com", "j.doe@acme.com"])
    assert probe.catch_all is False
    assert probe.results["jane@acme.com"].outcome == "accepted"
    assert probe.results["j.doe@acme.com"].outcome == "rejected"
    assert fake.rcpts[-1].endswith("@acme.com") and fake.rcpts[-1] not in ("jane@acme.com", "j.doe@acme.com")
    assert not fake.data_sent


def test_probe_detects_catch_all():
    probe = _prober(FakeSMTP(set(), catch_all=True)).probe("acme.com", ["mx"], ["jane@acme.com"])
    assert probe.catch_all is True


def test_probe_mail_from_rejected_is_blocked():
    probe = _prober(FakeSMTP({"jane@acme.com"}, mail_code=550)).probe("acme.com", ["mx"], ["jane@acme.com"])
    assert probe.results["jane@acme.com"].outcome == "blocked"
    assert probe.catch_all is None


def test_probe_connection_failure_is_error():
    probe = _prober(FakeSMTP(set(), fail_connect=True)).probe("acme.com", ["mx1", "mx2"], ["jane@acme.com"])
    assert probe.host is None
    assert probe.results["jane@acme.com"].outcome == "error"


def test_probe_greylisted_catch_all_test_is_inconclusive():
    fake = FakeSMTP({"jane@acme.com"})
    prober = _prober(fake)
    original = fake.rcpt

    def rcpt(addr):
        if addr == "jane@acme.com":
            return original(addr)
        return 451, b"4.7.1 greylisted"

    fake.rcpt = rcpt
    probe = prober.probe("acme.com", ["mx"], ["jane@acme.com"])
    assert probe.catch_all is None


# -- a real SMTP conversation over a socket, through smtplib ---------------------------


class _Handler(socketserver.StreamRequestHandler):
    mailboxes = {"owner@acme.test"}

    def handle(self):
        self.wfile.write(b"220 fake.mx ESMTP\r\n")
        while True:
            line = self.rfile.readline()
            if not line:
                return
            cmd = line.decode().strip()
            upper = cmd.upper()
            if upper.startswith("EHLO"):
                self.wfile.write(b"250-fake.mx\r\n250 SIZE 1000000\r\n")
            elif upper.startswith("MAIL FROM"):
                self.wfile.write(b"250 2.1.0 Ok\r\n")
            elif upper.startswith("RCPT TO"):
                addr = cmd.split(":", 1)[1].strip().strip("<>").split(">")[0]
                if addr in self.mailboxes:
                    self.wfile.write(b"250 2.1.5 Ok\r\n")
                else:
                    self.wfile.write(b"550 5.1.1 <" + addr.encode() + b">: Recipient address rejected: User unknown\r\n")
            elif upper.startswith("QUIT"):
                self.wfile.write(b"221 2.0.0 Bye\r\n")
                return
            elif upper.startswith("DATA"):
                raise AssertionError("verifier must never send DATA")
            else:
                self.wfile.write(b"502 5.5.2 Error: command not recognized\r\n")


@pytest.fixture
def smtp_server():
    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_real_smtp_session(smtp_server):
    prober = SmtpProber("mail.agency.test", "verify@agency.test", timeout=5, port=smtp_server)
    probe = prober.probe("acme.test", ["127.0.0.1"], ["owner@acme.test", "nobody@acme.test"])
    assert probe.host == "127.0.0.1"
    assert probe.catch_all is False
    assert probe.results["owner@acme.test"].outcome == "accepted"
    assert probe.results["nobody@acme.test"].outcome == "rejected"


def test_unreachable_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # nothing listening once closed
    prober = SmtpProber("h", "f@agency.test", timeout=2, port=port)
    probe = prober.probe("acme.test", ["127.0.0.1"], ["owner@acme.test"])
    assert probe.results["owner@acme.test"].outcome == "error"
