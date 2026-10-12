# LeadGen Agent: intent-based leads for SMMA and ads agencies

An autonomous agent that finds businesses that **asked for** social media management or ads
management help in the last 2–14 days. For each one it identifies the owner or decision-maker,
checks that person's email address strictly, and saves a CSV of warm leads.

```
 sources ──► date window ──► keyword prefilter ──► Claude intent check ──► decision-makers ──► strict verification ──► CSV / JSON
 (LinkedIn,   (2–14 days;     (drops sellers,       (buyer vs. seller,      (website crawl,     (syntax, role, disposable,
  Instagram,   undated posts   off-topic posts)      score, business,        JSON-LD, Claude,    MX, SMTP mailbox +
  Facebook, X,  dropped)                             poster, niche)          Hunter, patterns)   catch-all probe, verifier
  Reddit, web)                                                                                   API + dormancy)
        ▲
        └── niche research: Claude searches the web each week for niches with growth signals
            (e.g. newly opened gyms) and writes the queries used to find businesses in them
```

## What each requirement maps to

| Requirement | How it works | Code |
|---|---|---|
| **Intent-based triggering** | Searches for public posts with buyer-side phrases ("looking for a social media manager", "need someone to run Facebook ads", …) on LinkedIn, Instagram, Facebook, X, Craigslist and the open web, plus Reddit's API (`[Hiring]` posts only in r/forhire-style subs). Claude then decides whether the author is **buying** or **selling**, because most posts that mention "social media management" are agencies advertising themselves. | `queries.py`, `sources/`, `intent.py` |
| **Strict 2–14 day window** | Each post's date comes from the most reliable source available: the timestamp encoded in LinkedIn and X post IDs (exact to the millisecond), Reddit's API, feed dates, or the search engine's index date. Searches are restricted to the window on the API side too. **Posts without a trustworthy date are dropped.** | `dates.py`, `config.window` |
| **Smart audience targeting** | Once a week Claude runs web research (server-side web search tool) to find niches with live demand signals: openings, launches, funding, seasonal peaks. For each niche it writes search queries and subreddits. Posts found this way are judged as *growth signals*, kept separate from explicit requests, and have their own score threshold. | `niches.py` |
| **Decision-maker extraction** | Finds the business's own domain (from the post, or by search), then crawls the homepage plus the about, team and contact pages. It reads JSON-LD `founder` / `Person` data and has Claude extract named owners, founders, CEOs and marketing leads; a name is kept only if it appears verbatim on the page. Hunter.io and the company's email pattern are used when available. Titles are ranked (owner/founder > CEO > CMO/head of marketing > …), and assistants or support staff are excluded. **Role addresses (info@, support@, hello@, sales@ …) are never used.** | `enrich/` |
| **Strict email validation** | An address passes only when it is **positively confirmed** deliverable. Checks: syntax → suppression list → role account → disposable domain → MX records (null MX and dead domains rejected) → SMTP `RCPT TO` on the mailbox plus a random-address probe that detects **catch-all** domains → optional verification API (ZeroBounce / NeverBounce / MillionVerifier) → mailbox **activity** data, which rejects dormant inboxes. Catch-all, unknown, greylisted, full or inactive mailboxes are rejected. | `verify/` |
| **Structured output** | `output/leads_<timestamp>.csv` / `.json` plus a cumulative `leads_all.csv` (columns below). `rejected_<timestamp>.csv` gives the reason every post was dropped. | `output.py` |

### Output columns

`verified_email, contact_name, contact_role, business_name, business_website, platform,
intent_type, posted_date, posted_at, date_source, intent_url, services_needed, niche, location,
intent_score, summary, post_excerpt, email_source, verification, found_at`

* `platform`: where the intent was found (LinkedIn, Instagram, Facebook, X (Twitter), Reddit, …).
* `posted_date` / `posted_at`: when the person posted their need. `date_source` says how the date
  was established (`post_id` and `platform_api` are exact; `search_index` comes from the search
  engine and is accurate to the day).
* `intent_type`: `explicit_request` (they asked) or `growth_signal` (a researched-niche business showing need).
* `verification`: which checks confirmed the mailbox, e.g. `confirmed by smtp, zerobounce`.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp config.example.yaml config.yaml   # edit agency details, smtp_from, helo_host
cp .env.example .env                 # add API keys
```

| Key | Needed? | Used for |
|---|---|---|
| `ANTHROPIC_API_KEY` | **required** | Intent classification, niche research, people extraction (default model `claude-opus-5-5`) |
| `BRAVE_API_KEY` or `SERPER_API_KEY` | strongly recommended | Searching public posts with an exact date range. Without one, Claude's own web search tool is used. That needs no extra setup but costs much more per run. |
| `REDDIT_CLIENT_ID` / `_SECRET` | recommended | Reddit API at full rate limits (a free "script" app). Without it, slow anonymous access is used. |
| `ZEROBOUNCE_API_KEY` (or NeverBounce / MillionVerifier) | recommended, **required if you can't use port 25** | Second verification opinion. ZeroBounce also returns mailbox activity, used to reject dormant inboxes. |
| `HUNTER_API_KEY` | optional | Known decision-maker emails and the company's email pattern |

### SMTP verification needs port 25

The SMTP mailbox check opens a connection to the recipient's mail server on port 25. It sends
`MAIL FROM` / `RCPT TO` and then `QUIT`; it never sends a message. **Most cloud providers (AWS, GCP,
Azure) and home ISPs block outbound port 25.** You have two options:

1. Run the agent on a VPS that allows port 25 (several providers unblock it on request). Set
   `verification.smtp_from` to an address on a domain you own, and set `helo_host` to a hostname that
   resolves to that server, with matching reverse DNS. Otherwise many servers will refuse the probe.
2. Set `verification.smtp: false` and use a verification API key instead.

The best accuracy comes from both, with `confirmations_required: all`.

## Usage

```bash
leadgen run --limit 20          # first test: assess at most 20 new posts
leadgen run                     # full run with the window from config (default: last 14 days)
leadgen run --days 7            # only posts from the last 7 days
leadgen run --min-days 2        # skip posts newer than 2 days
leadgen run --every 6           # keep running autonomously, every 6 hours

leadgen niches --refresh        # re-run the niche research now and print it
leadgen verify jane@acme.com    # run the full verification on any address
leadgen suppress jane@acme.com acme.com   # never contact these (opt-outs, clients, competitors)
leadgen export                  # every lead delivered so far, as one CSV
```

Each run skips posts it has already processed. It never delivers the same email twice, or the same
business twice within 60 days. State lives in `data/leadgen.db` (SQLite).

To schedule with cron instead of `--every`:

```cron
15 */6 * * *  cd /path/to/hmida && .venv/bin/leadgen run >> output/cron.log 2>&1
```

You can also feed in posts from other tools you're licensed to use (a Sales Navigator export, a
saved-search tool, a VA's spreadsheet). List CSV / JSONL files under `sources.imports`, with the
columns `url, posted_at, text`, and optionally `platform, title, author, business_name, website`.

## Read this before relying on it

**No verifier can guarantee 100% deliverability.** No tool can, including the paid ones. A mailbox
can be deleted after it was checked, and some servers accept every address at `RCPT` and bounce
later. This agent is built to make bounces as rare as possible: it saves an address only when a
check *positively confirms* it, and rejects anything uncertain (catch-all, unknown, greylisted,
blocked, full, inactive). The cost is that some real leads are discarded, notably every business
whose domain is catch-all, and Yahoo/AOL addresses, which accept all recipients. If you use a
verification API, `verification.trust_provider_on_catch_all: true` lets the API alone confirm such
an address (its own "catch-all" answers are still rejected). Re-verify before
a campaign if a lead is more than a few weeks old (`leadgen verify`); results are cached for
`verification.cache_days`.

**LinkedIn and Instagram are reached through search engines, not by logging in.** Automated,
logged-in scraping breaks both platforms' terms of service, and they enforce it with account bans
and lawsuits. The agent finds public posts that Google or Brave have indexed and reads the indexed
title and snippet. That covers a meaningful share of public LinkedIn posts but less of Instagram,
and almost none of private Facebook groups. To add another data source you're allowed to use,
implement the small `Source` protocol in `sources/base.py`, or use the import source.

**Cold-email rules apply to what you do with the list.** In the US, CAN-SPAM requires accurate
headers, a physical address and a working opt-out. In the UK and EU (PECR / GDPR), B2B outreach to
corporate addresses usually relies on legitimate interest and needs an easy opt-out; sole traders
are treated as individuals. Canada's CASL is stricter. Add opt-outs with `leadgen suppress`, and
keep `data/` and `output/` private, because they contain personal data. This is not legal advice.

**Costs scale with volume.** Every post that passes the prefilter costs one Claude call (low
effort), and every qualified business costs one more for people extraction. That call sees only the
text around titles like "founder" or "owner", not whole pages. Searching through Claude's web search
tool, the fallback when no Brave or Serper key is set, makes about one research call per query per
run, so a search API key is much cheaper at scale. To change the model, set `llm.model` (for
example `claude-sonnet-5-5`).

## Development

```bash
pytest          # offline test suite: no network or API keys needed
ruff check src tests
```

The tests cover date decoding, every verification path (including a real SMTP conversation with a
local test server), the enrichment heuristics, source parsing, the request shapes the Claude SDK
sends, and an end-to-end run with fake sources.

Live integrations were written against the published API shapes of Brave Search, Serper,
Reddit, Hunter, ZeroBounce, NeverBounce and MillionVerifier, but have not been exercised against
those services. Run `leadgen run --limit 5 -v` first after adding a key.
