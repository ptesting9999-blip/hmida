#!/usr/bin/env bash
# Run the agent for free inside a Claude Code cloud session (or any Linux box with Docker
# but no port 25 / no DNS). Starts a private SearXNG search engine, then runs the agent with
# config.cloud.yaml. Usage:
#   scripts/run_in_cloud.sh            # doctor, then one run
#   scripts/run_in_cloud.sh --limit 20 # extra arguments go to `leadgen run`
#   scripts/run_in_cloud.sh doctor     # only check the setup
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG=${LEADGEN_CONFIG:-config.cloud.yaml}
PORT=8888
NAME=leadgen-searxng
WORK=data/searxng-cloud

log() { printf '\n== %s\n' "$*"; }

# -- Docker -------------------------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  log "starting the Docker daemon"
  (dockerd >/tmp/dockerd.log 2>&1 &)
  for _ in $(seq 1 30); do docker info >/dev/null 2>&1 && break; sleep 1; done
  docker info >/dev/null 2>&1 || { echo "Docker is not available (see /tmp/dockerd.log)"; exit 1; }
fi

# -- SearXNG settings: reach the internet through this machine's proxy, if it has one ------
mkdir -p "$WORK"
cp searxng/settings.yml "$WORK/settings.yml"
{
  printf '\n# added by scripts/run_in_cloud.sh\n'
  proxy=${HTTPS_PROXY:-${https_proxy:-}}
  if [ -n "$proxy" ]; then
    ca=${SSL_CERT_FILE:-${REQUESTS_CA_BUNDLE:-}}
    printf 'outgoing:\n  request_timeout: 6.0\n  max_request_timeout: 15.0\n'
    printf '  proxies:\n    all://:\n      - %s\n' "$proxy"
    if [ -n "$ca" ] && [ -f "$ca" ]; then
      cp "$ca" "$WORK/proxy-ca.crt"
      printf '  verify: /etc/searxng/proxy-ca.crt\n'
    fi
  fi
} >>"$WORK/settings.yml"
# The appended outgoing block replaces the one copied from searxng/settings.yml.
python3 - "$WORK/settings.yml" <<'EOF'
import re
import sys
path = sys.argv[1]
text = open(path).read()
head, sep, tail = text.partition("# added by scripts/run_in_cloud.sh")
if "outgoing:" in tail:
    # the block's own lines are indented (or blank); the next top-level key ends it
    head = re.sub(r"(?m)^outgoing:.*\n(?:[ \t]+.*\n|[ \t]*\n)*", "", head)
open(path, "w").write(head + sep + tail)
EOF

# -- SearXNG container --------------------------------------------------------------------
# Settings may have changed since the container started: always start a fresh one.
docker rm -f "$NAME" >/dev/null 2>&1 || true
if ! curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  log "starting SearXNG on 127.0.0.1:$PORT"
  image=searxng/searxng:latest
  docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image" >/dev/null 2>&1 || {
    image=mirror.gcr.io/searxng/searxng:latest   # Docker Hub rate-limits shared IPs
    docker image inspect "$image" >/dev/null 2>&1 || docker pull -q "$image" >/dev/null
  }
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  ca_env=()
  if [ -f "$WORK/proxy-ca.crt" ]; then
    # SearXNG's HTTP client takes its trusted CAs from SSL_CERT_FILE, not outgoing.verify;
    # without this every engine fails with CertificateVerifyError behind the proxy.
    ca_env=(-e SSL_CERT_FILE=/etc/searxng/proxy-ca.crt -e REQUESTS_CA_BUNDLE=/etc/searxng/proxy-ca.crt
            -e CURL_CA_BUNDLE=/etc/searxng/proxy-ca.crt)
  fi
  docker run -d --name "$NAME" --network host \
    -v "$PWD/$WORK:/etc/searxng" \
    "${ca_env[@]}" \
    -e GRANIAN_HOST=127.0.0.1 -e GRANIAN_PORT=$PORT \
    -e SEARXNG_SECRET="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')" \
    "$image" >/dev/null
  for _ in $(seq 1 30); do
    curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -sf --noproxy '*' "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 || {
    echo "SearXNG did not start:"; docker logs --tail 20 "$NAME"; exit 1;
  }
fi

# -- the agent ----------------------------------------------------------------------------
if [ ! -x .venv/bin/leadgen ]; then
  log "installing the agent"
  python3 -m venv .venv
  .venv/bin/pip install -q -e .
fi
export NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost" no_proxy="${no_proxy:-},127.0.0.1,localhost"

log "checking the setup"
doctor_ok=0
.venv/bin/leadgen doctor -c "$CONFIG" || doctor_ok=$?
[ "${1:-}" = "doctor" ] && exit $doctor_ok
if [ $doctor_ok -ne 0 ]; then
  echo
  echo "Fix the FAIL lines above first (environment variables go in the cloud environment's"
  echo "settings, never in chat or in this repository), then run this script again."
  exit $doctor_ok
fi

log "running the agent"
.venv/bin/leadgen run -c "$CONFIG" "$@"
echo
echo "Leads: output/leads_all.csv   Why posts were dropped: output/rejected_*.csv"
