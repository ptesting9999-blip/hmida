"""`leadgen doctor`: check that keys, DNS and port 25 work before the first real run.

Every check is cheap: no search credits, verification credits or Claude tokens are spent,
except one search query when a Brave / Serper key is configured.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import anthropic
import dns.exception
import dns.resolver
import requests

from leadgen.config import Config, Secrets
from leadgen.llm import has_credentials
from leadgen.verify.dns_check import MxResult, lookup_mx
from leadgen.verify.verifier import DNS_CANARY_DOMAIN

Status = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    fix: str = ""


ICONS = {"ok": "OK  ", "warn": "WARN", "fail": "FAIL"}


def format_checks(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"[{ICONS[c.status]}] {c.name}: {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"       fix: {c.fix}")
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    lines.append("")
    lines.append("Ready to run." if not fails else f"{fails} problem(s) must be fixed before running.")
    if warns:
        lines.append(f"{warns} warning(s): the agent works, with reduced coverage or accuracy.")
    return "\n".join(lines)


def _http_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"HTTP {exc.response.status_code}"
    return exc.__class__.__name__


def _is_auth_error(exc: Exception) -> bool:
    return isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code in (401, 403)


class Doctor:
    def __init__(
        self,
        cfg: Config,
        secrets: Secrets,
        *,
        session: Optional[requests.Session] = None,
        claude: Optional[anthropic.Anthropic] = None,
        mx_lookup: Callable[[str], MxResult] = lookup_mx,
        connect: Callable[[str, int, float], bytes] | None = None,
        resolve_host: Callable[[str], bool] | None = None,
    ) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", cfg.http.user_agent)
        self._claude = claude
        self.mx_lookup = mx_lookup
        self.connect = connect or _smtp_banner
        self.resolve_host = resolve_host or _resolves

    def run(self) -> list[Check]:
        checks = [self.claude(), self.search(), self.reddit(), self.hunter()]
        checks += self.verification()
        checks.append(self.output_dir())
        return checks

    def _get(self, url: str, **kw: Any) -> Any:
        resp = self.session.get(url, timeout=15, **kw)
        resp.raise_for_status()
        return resp.json()

    # -- Claude ------------------------------------------------------------------------

    def claude(self) -> Check:
        name = "Claude API"
        try:
            client = self._claude or anthropic.Anthropic(api_key=self.secrets.anthropic_api_key, max_retries=1)
            if self._claude is None and not has_credentials(client):
                return Check(name, "fail", "no Anthropic credentials found", "set ANTHROPIC_API_KEY in .env")
            model = client.models.retrieve(self.cfg.llm.model)
        except anthropic.AuthenticationError:
            return Check(name, "fail", "the API key was rejected", "set a valid ANTHROPIC_API_KEY in .env")
        except anthropic.NotFoundError:
            return Check(name, "fail", f"model {self.cfg.llm.model} is not available to this key", "change llm.model in config.yaml")
        except anthropic.APIConnectionError:
            return Check(name, "fail", "could not reach api.anthropic.com", "check this machine's internet access")
        except anthropic.AnthropicError as exc:
            return Check(name, "fail", str(exc)[:200], "set ANTHROPIC_API_KEY in .env")
        return Check(name, "ok", f"key works, model {model.id} available")

    # -- discovery ---------------------------------------------------------------------

    def search(self) -> Check:
        name = "Post search"
        s = self.secrets
        choice = self.cfg.sources.search.provider
        try:
            if choice in ("auto", "brave") and s.brave_api_key:
                self._get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": "social media manager", "count": 1},
                    headers={"X-Subscription-Token": s.brave_api_key, "Accept": "application/json"},
                )
                return Check(name, "ok", "Brave Search key works")
            if choice in ("auto", "serper") and s.serper_api_key:
                resp = self.session.post(
                    "https://google.serper.dev/search", json={"q": "social media manager", "num": 1},
                    headers={"X-API-KEY": s.serper_api_key}, timeout=15,
                )
                resp.raise_for_status()
                return Check(name, "ok", "Serper key works")
        except (requests.RequestException, ValueError) as exc:
            return Check(name, "fail", f"search API check failed ({_http_error(exc)})", "check BRAVE_API_KEY / SERPER_API_KEY")
        if choice in ("brave", "serper"):
            return Check(name, "fail", f"provider {choice} selected but no key set", f"set {choice.upper()}_API_KEY")
        return Check(
            name, "warn", "no Brave/Serper key: Claude's web search tool will be used (works, but costs more per run)",
            "add BRAVE_API_KEY or SERPER_API_KEY to cut search cost",
        )

    def reddit(self) -> Check:
        name = "Reddit"
        if not self.cfg.sources.reddit.enabled:
            return Check(name, "ok", "disabled in config")
        s = self.secrets
        if not (s.reddit_client_id and s.reddit_client_secret):
            return Check(
                name, "warn", "no app credentials: slow anonymous access, which Reddit often blocks from servers",
                "create a free 'script' app at https://www.reddit.com/prefs/apps and set REDDIT_CLIENT_ID / _SECRET",
            )
        try:
            resp = self.session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=(s.reddit_client_id, s.reddit_client_secret),
                data={"grant_type": "client_credentials"}, timeout=15,
            )
            resp.raise_for_status()
            if "access_token" not in resp.json():
                raise ValueError("no access_token in response")
        except (requests.RequestException, ValueError) as exc:
            return Check(name, "fail", f"credentials rejected ({_http_error(exc)})", "check REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET")
        return Check(name, "ok", "app credentials work")

    # -- enrichment --------------------------------------------------------------------

    def hunter(self) -> Check:
        name = "Hunter.io"
        if not self.secrets.hunter_api_key or self.cfg.enrichment.use_hunter == "never":
            status: Status = "fail" if self.cfg.enrichment.use_hunter == "always" else "ok"
            return Check(name, status, "not configured (optional)", "set HUNTER_API_KEY")
        try:
            data = self._get("https://api.hunter.io/v2/account", params={"api_key": self.secrets.hunter_api_key})
        except (requests.RequestException, ValueError) as exc:
            if _is_auth_error(exc):
                return Check(name, "fail", "key rejected", "check HUNTER_API_KEY")
            return Check(name, "warn", f"could not confirm the key ({_http_error(exc)})")
        plan = (data.get("data") or {}).get("plan_name") or "unknown plan"
        return Check(name, "ok", f"key works ({plan})")

    # -- verification ------------------------------------------------------------------

    def verification(self) -> list[Check]:
        v = self.cfg.verification
        checks: list[Check] = []
        provider = self._provider_check()
        if provider:
            checks.append(provider)

        mx = self.mx_lookup(DNS_CANARY_DOMAIN)
        dns_ok = mx.status == "ok"
        checks.append(
            Check("DNS resolver", "ok", "MX lookups work")
            if dns_ok
            else Check("DNS resolver", "fail", f"cannot resolve MX for {DNS_CANARY_DOMAIN} ({mx.detail})",
                       "this machine needs working DNS; every address would come back 'unknown'")
        )

        smtp_ok = False
        if v.smtp:
            checks += self._smtp_identity()
            if dns_ok:
                host = mx.hosts[0]
                try:
                    banner = self.connect(host, v.smtp_port, min(v.smtp_timeout, 10.0))
                    smtp_ok = banner.startswith(b"220")
                    detail = f"connected to {host}:{v.smtp_port}" if smtp_ok else f"unexpected greeting {banner[:60]!r}"
                except OSError as exc:
                    detail = f"cannot connect to {host}:{v.smtp_port} ({exc.__class__.__name__}) - outbound port 25 is blocked here"
                checks.append(
                    Check("SMTP port 25", "ok" if smtp_ok else "fail", detail,
                          "run on a server that allows outbound port 25, or set verification.smtp: false and use a verification API")
                )

        provider_ok = provider is not None and provider.status == "ok"
        if not smtp_ok and provider is not None and provider.status == "warn":
            checks.append(Check("Email verification", "warn", f"depends entirely on {provider.name}, whose key could not be confirmed",
                                "re-run doctor once the verification API is reachable"))
        elif not smtp_ok and not provider_ok:
            checks.append(Check("Email verification", "fail", "no working way to verify mailboxes: every lead would be rejected",
                                "fix port 25 access or add ZEROBOUNCE_API_KEY / NEVERBOUNCE_API_KEY / MILLIONVERIFIER_API_KEY"))
        else:
            methods = " + ".join(m for m, ok in (("SMTP", smtp_ok), (provider.name if provider else "", provider_ok)) if ok)
            checks.append(Check("Email verification", "ok", f"mailboxes will be verified via {methods}"))
        return checks

    def _smtp_identity(self) -> list[Check]:
        v = self.cfg.verification
        out = []
        if v.smtp_from.endswith(("@example.com", "@yourdomain.com")):
            out.append(Check("SMTP sender", "warn", f"smtp_from is still {v.smtp_from}",
                             "set verification.smtp_from to an address on a domain you own (with SPF)"))
        else:
            out.append(Check("SMTP sender", "ok", v.smtp_from))
        if v.helo_host in ("localhost", "mail.yourdomain.com") or not self.resolve_host(v.helo_host):
            out.append(Check("SMTP HELO name", "warn", f"helo_host {v.helo_host!r} does not resolve publicly",
                             "set verification.helo_host to this server's hostname (A record + matching reverse DNS)"))
        else:
            out.append(Check("SMTP HELO name", "ok", v.helo_host))
        return out

    def _provider_check(self) -> Optional[Check]:
        s, choice = self.secrets, self.cfg.verification.provider
        if choice == "none":
            return None
        candidates = [
            ("ZeroBounce", "zerobounce", s.zerobounce_api_key, self._zerobounce),
            ("NeverBounce", "neverbounce", s.neverbounce_api_key, self._neverbounce),
            ("MillionVerifier", "millionverifier", s.millionverifier_api_key, self._millionverifier),
        ]
        for label, key_name, key, fn in candidates:
            if choice not in ("auto", key_name):
                continue
            if not key:
                if choice == key_name:
                    return Check(label, "fail", "selected but no key set", f"set {key_name.upper()}_API_KEY")
                continue
            try:
                status, detail = fn(key)
            except (requests.RequestException, ValueError) as exc:
                if _is_auth_error(exc):
                    status, detail = "fail", "key rejected"
                else:
                    status, detail = "warn", f"could not confirm the key or credits ({_http_error(exc)})"
            return Check(label, status, detail, f"check {key_name.upper()}_API_KEY and your credit balance")
        return None

    def _zerobounce(self, key: str) -> tuple[Status, str]:
        credits = float(self._get("https://api.zerobounce.net/v2/getcredits", params={"api_key": key}).get("Credits", -1))
        if credits < 0:
            return "fail", "key rejected"
        return ("ok" if credits > 0 else "fail"), f"{credits:.0f} credits left"

    def _neverbounce(self, key: str) -> tuple[Status, str]:
        data = self._get("https://api.neverbounce.com/v4/account/info", params={"key": key})
        if data.get("status") != "success":
            return "fail", f"key rejected ({data.get('message') or data.get('status')})"
        info = data.get("credits_info") or {}
        left = sum(int(info.get(k) or 0) for k in ("paid_credits_remaining", "free_credits_remaining"))
        return ("ok" if left > 0 else "fail"), f"{left} credits left"

    def _millionverifier(self, key: str) -> tuple[Status, str]:
        data = self._get("https://api.millionverifier.com/api/v3/credits", params={"api": key})
        if data.get("error"):
            return "fail", f"key rejected ({data['error']})"
        credits = int(data.get("credits") or 0)
        return ("ok" if credits > 0 else "fail"), f"{credits} credits left"

    # -- local -------------------------------------------------------------------------

    def output_dir(self) -> Check:
        try:
            for d in (Path(self.cfg.output.dir), Path(self.cfg.db_path).parent):
                d.mkdir(parents=True, exist_ok=True)
                probe = d / ".write_test"
                probe.write_text("ok")
                probe.unlink()
        except OSError as exc:
            return Check("Output folders", "fail", f"not writable: {exc}", "fix permissions or change output.dir / db_path")
        return Check("Output folders", "ok", f"{self.cfg.output.dir}/ and {self.cfg.db_path} writable")


def _smtp_banner(host: str, port: int, timeout: float) -> bytes:
    """Open a TCP connection to a mail server, read its greeting, say QUIT. Sends nothing else."""
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        banner = sock.recv(512)
        try:
            sock.sendall(b"QUIT\r\n")
        except OSError:
            pass
        return banner


def _resolves(host: str) -> bool:
    try:
        dns.resolver.resolve(host, "A", lifetime=5)
        return True
    except dns.exception.DNSException:
        return False
