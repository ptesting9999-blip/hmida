"""Offline email checks: syntax, role accounts, disposable and free-mail domains."""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Optional

# Pragmatic RFC 5322 subset: what real mailboxes use, not every legal oddity.
_LOCAL_RE = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*$", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^(?=.{4,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}$", re.IGNORECASE)

# Compared after lowercasing and removing digits and . _ - separators ("customer-care2" -> "customercare").
ROLE_ACCOUNTS = frozenset(
    """
    info information contact contactus hello hi hey support help helpdesk admin administrator office team
    sales enquiries enquiry inquiries inquiry mail email marketing media press pr news newsletter billing
    accounts accounting finance invoice invoices payments careers jobs job hr recruiting recruitment
    noreply donotreply postmaster hostmaster webmaster abuse security privacy legal compliance service
    services customerservice customersupport customercare care feedback orders order shop store booking
    bookings book reservations reservation reception frontdesk studio gym clinic events partners
    partnerships bizdev general all everyone staff social socialmedia ads advertising it tech ops
    operations root sysadmin test demo welcome reply notifications alerts membership members community
    wholesale collab collabs hire hiring admissions enquire contacto kontakt bonjour hallo hola ciao
    ventas reservas anfragen appointments appointment frontoffice frontdesk mailerdaemon daemon
    accountspayable payable receivable salesteam infodesk helpline
    """.split()
)
# Generic words that mark a qualified inbox: info.uk@, sales-london@, bookings.miami@ (word first)
# or nyc.info@, acme.sales@acme.com (word last, after a short code or the company's own name).
# jane.sales@ stays personal.
STRONG_ROLE_WORDS = frozenset(
    """
    info information support sales contact contactus enquiries enquiry inquiries inquiry hello
    helpdesk bookings booking orders reception noreply donotreply reservations careers billing
    """.split()
)
# A role word glued to one of these is still generic: salesteam@, marketingdept@, contactme@.
_ROLE_SUFFIXES = ("team", "desk", "dept", "department", "group", "office", "hq", "me", "us", "mail", "line")
_GENERIC_PREFIXES = ("info", "contact", "support", "enquir", "inquir", "hello", "booking", "reservation")

FREE_PROVIDERS = frozenset(
    """
    gmail.com googlemail.com yahoo.com yahoo.co.uk yahoo.ca yahoo.com.au ymail.com rocketmail.com
    hotmail.com hotmail.co.uk hotmail.fr outlook.com live.com msn.com aol.com icloud.com me.com mac.com
    proton.me protonmail.com pm.me gmx.com gmx.net gmx.de web.de yandex.com yandex.ru mail.com zohomail.com
    mail.ru hey.com fastmail.com tutanota.com qq.com 163.com
    """.split()
)


def split_email(email: str) -> tuple[str, str]:
    local, _, domain = email.strip().rpartition("@")
    return local, domain.lower()


def check_syntax(email: str) -> Optional[str]:
    """Return a problem description, or None if the address is well formed."""
    email = email.strip()
    if email.count("@") != 1:
        return "must contain exactly one @"
    local, domain = split_email(email)
    if not local or len(local) > 64:
        return "local part empty or longer than 64 characters"
    if len(email) > 254:
        return "address longer than 254 characters"
    if not _LOCAL_RE.match(local):
        return "invalid characters or dots in local part"
    if not _DOMAIN_RE.match(domain):
        return "invalid domain"
    return None


def normalized_local(local: str) -> str:
    return re.sub(r"[\d._\-+]", "", local.lower())


def is_role_account(email: str) -> bool:
    """Generic inboxes rather than a person: info@, support2@, customer-care@, sales.team@,
    info.uk@, salesteam@, contactme@, hola@ ..."""
    local, domain = split_email(email)
    base = local.lower().split("+", 1)[0]
    joined = normalized_local(base)
    if joined in ROLE_ACCOUNTS or joined.startswith(_GENERIC_PREFIXES):
        return True
    if any(joined == word + suffix for word in ROLE_ACCOUNTS for suffix in _ROLE_SUFFIXES):
        return True
    tokens = [t for t in (re.sub(r"\d", "", part) for part in re.split(r"[._\-]", base)) if t]
    if not tokens:
        return False
    if all(t in ROLE_ACCOUNTS for t in tokens) or tokens[0] in STRONG_ROLE_WORDS:
        return True
    brand = domain.split(".")[0]
    return tokens[-1] in STRONG_ROLE_WORDS and all(len(t) <= 3 or t == brand for t in tokens[:-1])


def is_free_provider(email_or_domain: str) -> bool:
    domain = split_email(email_or_domain)[1] if "@" in email_or_domain else email_or_domain.lower()
    return domain in FREE_PROVIDERS


def load_disposable_domains(extra_file: Optional[str] = None) -> frozenset[str]:
    text = resources.files("leadgen").joinpath("data/disposable_domains.txt").read_text()
    lines = text.splitlines()
    if extra_file:
        lines += Path(extra_file).read_text().splitlines()
    return frozenset(line.strip().lower() for line in lines if line.strip() and not line.startswith("#"))


def is_disposable(email: str, disposable: frozenset[str]) -> bool:
    domain = split_email(email)[1]
    parts = domain.split(".")
    # match the domain and its parents (x.mailinator.com -> mailinator.com)
    return any(".".join(parts[i:]) in disposable for i in range(len(parts) - 1))
