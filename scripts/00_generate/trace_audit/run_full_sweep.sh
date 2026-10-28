#!/usr/bin/env bash
# Full grounding audit over the 10,236-trace pool.
#
# Prompt v2, gemini-3.7-flash, thinking low, temperature 0, concurrency 8,
# four shards run one after the other (never in parallel, that buys HTTP 429s).
#
# Resumable. Every finished item is fsynced to audit_grounding.jsonl as its own
# line, and the runner skips ids that are already in that file. So if this dies,
# rerun exactly the same command and it continues where it stopped.
#
# Pass 1 does the sweep. Passes 2 and 3 add --retry-errored, which drops stored
# records whose status is not ok and does them again. A copy of the file is made
# before each of those passes.
#
# The API key is never stored in this directory. It comes from the environment,
# or from a mode-600 file in the home directory.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/audit_grounding.jsonl"
LOG="$HERE/audit.log"
PROMPT="$HERE/prompt_grounding_v2.txt"
CORPUS="$HERE/corpus.jsonl"
CONC=8
TOTAL=10236
# Optional mode-600 shell fragment that exports GEMINI_API_KEY. Never inside the repo.
KEYFILE="${GEMINI_KEYFILE:-$HOME/.gemini_audit_key}"

cd "$HERE" || exit 1

if [ -z "${GEMINI_API_KEY:-}" ] && [ -f "$KEYFILE" ]; then
  # shellcheck disable=SC1090
  . "$KEYFILE"
fi
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEY is not set and $KEYFILE does not exist" >&2
  exit 1
fi
export GEMINI_API_KEY

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== sweep driver start, pid $$ ==="
say "corpus=$CORPUS ($TOTAL items) prompt=$(basename "$PROMPT") concurrency=$CONC"
say "out=$OUT"

for pass in 1 2 3; do
  if [ "$pass" -gt 1 ]; then
    if [ -f "$OUT" ]; then cp -f "$OUT" "$OUT.bak.pass$pass"; fi
    EXTRA="--retry-errored"
    say "--- pass $pass: redo errored records. backup at $OUT.bak.pass$pass ---"
  else
    EXTRA=""
    say "--- pass $pass: first sweep ---"
  fi
  for s in 0 1 2 3; do
    say ">>> pass $pass shard $s/4 starting"
    python3 "$HERE/audit_traces_gemini.py" \
      --corpus "$CORPUS" \
      --prompt "$PROMPT" \
      --out "$OUT" \
      --shard "$s/4" \
      --concurrency "$CONC" \
      --total "$TOTAL" \
      --log-every 100 \
      $EXTRA >> "$LOG" 2>&1
    rc=$?
    have=$(wc -l < "$OUT" 2>/dev/null || echo 0)
    say "<<< pass $pass shard $s/4 exit $rc | records on disk $have / $TOTAL"
    sleep 60
  done
done

have=$(wc -l < "$OUT" 2>/dev/null || echo 0)
say "=== sweep driver done | records $have / $TOTAL ==="
say "aggregate report: python3 $HERE/aggregate_report.py"
