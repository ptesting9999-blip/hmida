"""Command line: leadgen run | niches | verify | reverify | suppress | export | doctor"""

from __future__ import annotations

import argparse
import logging
import signal
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

from leadgen.compliance import Suppression
from leadgen.config import Config, Secrets, load_config, load_dotenv
from leadgen.dates import utcnow
from leadgen.doctor import Doctor, format_checks
from leadgen.factory import SetupError, build_agent, make_http, make_llm, make_verifier
from leadgen.llm import LLMUnavailable
from leadgen.niches import NicheResearcher
from leadgen.maintenance import reverify_leads
from leadgen.output import EXPORT_COLUMNS, write_csv
from leadgen.store import Store

log = logging.getLogger("leadgen")


def _config(path: Optional[str]) -> Config:
    if path is None and Path("config.yaml").is_dir():
        raise SetupError("./config.yaml is a folder, not a file (Docker creates one when the file is missing): "
                         "remove it and copy config.example.yaml to config.yaml")
    if path is None and Path("config.yaml").is_file():
        path = "config.yaml"
    return load_config(path)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    if args.days is not None:
        cfg.window.max_age_days = args.days
    if args.min_days is not None:
        cfg.window.min_age_days = args.min_days
    cfg = Config.model_validate(cfg.model_dump())  # re-validate overrides
    agent = build_agent(cfg, Secrets.from_env())
    while True:
        if not args.every:
            print(agent.run(limit=args.limit).summary())
            return 0
        try:
            print(agent.run(limit=args.limit).summary(), flush=True)
        except LLMUnavailable as exc:
            # Stay up and try again next cycle: exiting would make Docker restart us in a tight
            # loop. The preflight check fails before any paid search runs, so this costs nothing.
            log.error("Claude unavailable (%s) - fix ANTHROPIC_API_KEY; retrying next cycle", exc)
        except Exception:
            # Network blips etc. must not stop an unattended agent; try again next cycle.
            log.exception("run failed; retrying at the next scheduled run")
        log.info("next run in %.1f hours (Ctrl+C to stop)", args.every)
        time.sleep(args.every * 3600)


def cmd_niches(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    secrets = Secrets.from_env()
    store = Store(cfg.db_path)
    niches = NicheResearcher(cfg, make_llm(cfg, secrets), store).get(refresh=args.refresh)
    if not niches:
        print("No niches yet (research disabled or returned nothing).")
    for n in niches:
        print(f"\n## {n.name}\n{n.rationale}")
        for q in n.search_queries:
            print(f"  query: {q}")
        if n.subreddits:
            print(f"  subreddits: {', '.join('r/' + s for s in n.subreddits)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    _, api_http = make_http(cfg)
    verifier = make_verifier(cfg, Secrets.from_env(), None, api_http)
    worst = 0
    for email in args.emails:
        r = verifier.verify(email)
        print(f"{email}: {r.status.upper()} - {r.reason}")
        for k, v in r.checks.items():
            print(f"    {k}: {v}")
        worst = max(worst, 0 if r.deliverable else 1)
    return worst


def cmd_suppress(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    s = Suppression(cfg.suppression_file)
    for entry in args.entries:
        s.add(entry)
        print(f"suppressed {entry}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    rows = Store(cfg.db_path).all_leads(include_inactive=args.all)
    if not args.all:
        rows = Suppression(cfg.suppression_file).filter_rows(rows)  # opt-outs never leave the building
    out = Path(args.out or f"{cfg.output.dir}/all_leads_{utcnow():%Y%m%d}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows, EXPORT_COLUMNS)
    print(f"exported {len(rows)} leads to {out}")
    return 0


def cmd_reverify(args: argparse.Namespace) -> int:
    cfg = _config(args.config)
    store = Store(cfg.db_path)
    _, api_http = make_http(cfg)
    verifier = make_verifier(cfg, Secrets.from_env(), store, api_http)
    report = reverify_leads(store, verifier, args.older_than, cfg.verification.greylist_retry_seconds)
    print(report.summary())
    rows = Suppression(cfg.suppression_file).filter_rows(store.all_leads())
    out = Path(f"{cfg.output.dir}/verified_leads_{utcnow():%Y%m%d-%H%M%S}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows, EXPORT_COLUMNS)
    print(f"wrote {len(rows)} currently verified leads to {out}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    checks = Doctor(_config(args.config), Secrets.from_env()).run()
    print(format_checks(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="YAML config (default: ./config.yaml if present)")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    p = argparse.ArgumentParser(prog="leadgen", description="Intent-based lead generation agent for SMMA / ads agencies")
    sub = p.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", parents=[common], help="check keys, DNS and port 25 before the first run")
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run", parents=[common], help="find, qualify, enrich and verify new leads")
    run.add_argument("--days", type=int, help="max post age in days (overrides window.max_age_days)")
    run.add_argument("--min-days", type=int, help="min post age in days (overrides window.min_age_days)")
    run.add_argument("--limit", type=int, help="assess at most N new posts (useful for a first test)")
    run.add_argument("--every", type=float, metavar="HOURS", help="keep running, once every HOURS")
    run.set_defaults(func=cmd_run)

    niches = sub.add_parser("niches", parents=[common], help="show (or refresh) the niches the agent is targeting")
    niches.add_argument("--refresh", action="store_true", help="re-run the web research now")
    niches.set_defaults(func=cmd_niches)

    verify = sub.add_parser("verify", parents=[common], help="run the strict verification on email addresses")
    verify.add_argument("emails", nargs="+")
    verify.set_defaults(func=cmd_verify)

    suppress = sub.add_parser("suppress", parents=[common], help="never contact these emails / domains")
    suppress.add_argument("entries", nargs="+", help="jane@acme.com or acme.com")
    suppress.set_defaults(func=cmd_suppress)

    export = sub.add_parser("export", parents=[common], help="export every lead delivered so far to CSV")
    export.add_argument("--out", help="output CSV path")
    export.add_argument("--all", action="store_true", help="include leads whose email no longer verifies")
    export.set_defaults(func=cmd_export)

    reverify = sub.add_parser("reverify", parents=[common], help="re-check stored leads' emails before a campaign")
    reverify.add_argument("--older-than", type=int, default=7, metavar="DAYS", help="re-check leads last verified more than DAYS ago (default 7)")
    reverify.set_defaults(func=cmd_reverify)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("urllib3", "httpx", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_dotenv()
    # `docker stop` sends SIGTERM: turn it into SystemExit so cleanup (writing lead files) runs.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        return args.func(args)
    except (SetupError, LLMUnavailable, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (sqlite3.Error, PermissionError) as exc:
        print(
            f"error: cannot write the database or output files ({exc}). If you run in Docker, make the "
            "mounted folders writable by the container user: sudo chown -R 1000:1000 data output",
            file=sys.stderr,
        )
        return 2
    except KeyboardInterrupt:
        return 130
