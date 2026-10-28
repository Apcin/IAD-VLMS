#!/usr/bin/env python3
"""Wall clock and token volume for the full 10,236-item sweep.

Token means are taken from the 200-item calibration pass, which is 100 normal /
100 anomalous. The real pool is 4,562 normal / 5,674 anomalous, so the per-class
means are re-weighted to the pool before extrapolating.
"""
# Calibration inputs (calib_200.jsonl, calib_200_pass1.jsonl, calib_repeat_40_pass2.jsonl,
# calib_rerun50_v2_c16.jsonl) are NOT shipped in this repository, see README.md in this
# directory. Regenerate them with make_calib_sample.py + audit_traces_gemini.py first.
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_POOL, N_NORM, N_ANOM = 10236, 4562, 5674
W_N, W_A = N_NORM / N_POOL, N_ANOM / N_POOL

p1 = [json.loads(l) for l in open(HERE / "calib_200_pass1.jsonl")]
ok = [r for r in p1 if r["status"] == "ok"]
norm = [r for r in ok if r["gold_label"] == "no"]
anom = [r for r in ok if r["gold_label"] == "yes"]

# v2 adds the counts array, so output grows. Measure the growth on the paired 50.
v1 = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "calib_200_pass1.jsonl")}
v2 = {json.loads(l)["id"]: json.loads(l) for l in open(HERE / "calib_rerun50_v2_c16.jsonl")}
pair = [i for i in v2 if v2[i]["status"] == "ok" and v1.get(i, {}).get("status") == "ok"]
o1 = st.mean([v1[i]["output_tokens"] for i in pair])
o2 = st.mean([v2[i]["output_tokens"] for i in pair])
t1 = st.mean([v1[i].get("thought_tokens") or 0 for i in pair])
t2 = st.mean([v2[i].get("thought_tokens") or 0 for i in pair])
i1 = st.mean([v1[i]["prompt_tokens"] for i in pair])
i2 = st.mean([v2[i]["prompt_tokens"] for i in pair])
print(f"paired 50 items, v1 vs v2 prompt")
print(f"  input   {i1:7.0f} -> {i2:7.0f}   ({100*(i2/i1-1):+.1f} %)")
print(f"  output  {o1:7.0f} -> {o2:7.0f}   ({100*(o2/o1-1):+.1f} %)")
print(f"  think   {t1:7.0f} -> {t2:7.0f}")
in_scale, out_scale = i2 / i1, (o2 + t2) / (o1 + t1)

print("\n--- per-item means from the 200-item pass (prompt v1) ---")
for name, sub in (("normal", norm), ("anomalous", anom)):
    print(f"  {name:10s} in {st.mean([r['prompt_tokens'] for r in sub]):7.0f}  "
          f"out {st.mean([r['output_tokens'] for r in sub]):5.0f}  "
          f"think {st.mean([r.get('thought_tokens') or 0 for r in sub]):5.0f}  n={len(sub)}")

IN = W_N * st.mean([r["prompt_tokens"] for r in norm]) + W_A * st.mean([r["prompt_tokens"] for r in anom])
OUT = W_N * st.mean([r["output_tokens"] for r in norm]) + W_A * st.mean([r["output_tokens"] for r in anom])
THK = W_N * st.mean([r.get("thought_tokens") or 0 for r in norm]) + \
      W_A * st.mean([r.get("thought_tokens") or 0 for r in anom])
print(f"\npool-weighted per item (v1): in {IN:.0f}  out {OUT:.0f}  think {THK:.0f}  "
      f"billed-out {OUT+THK:.0f}")
INv2, OUTv2 = IN * in_scale, (OUT + THK) * out_scale
print(f"pool-weighted per item (v2): in {INv2:.0f}  billed-out {OUTv2:.0f}")

print(f"\n--- token volume for all {N_POOL} items, prompt v2, one pass ---")
print(f"  input tokens  : {INv2*N_POOL/1e6:8.1f} M")
print(f"  output tokens : {OUTv2*N_POOL/1e6:8.1f} M   (answer + thinking, both billed as output)")
print(f"  total tokens  : {(INv2+OUTv2)*N_POOL/1e6:8.1f} M")

print("\n--- measured wall clock (each row is a real run, not a model) ---")
RUNS = [("v1  200 items  c=8   (runner still had the truncation bug)", 200, 17.1),
        ("v1   40 items  c=8   (repeat pass, bug fixed)",               40,  2.4),
        ("v2   50 items  c=16  (anomaly-enriched set)",                 50,  3.3),
        ("v2   50 items  c=24  (anomaly-enriched set)",                 50,  7.0),
        ("v2   50 items  c=8   (quota wall, 23/50 gave up on HTTP 429)", 50, 13.2)]
for name, n, mins in RUNS:
    sec = mins * 60 / n
    print(f"  {name:58s} {sec:5.2f} s/item   {N_POOL*sec/3600:6.1f} h for {N_POOL}")

print("\n--- extrapolation to 10,236 items ---")
print("  Concurrency is NOT the binding constraint. The shared Vertex quota is.")
print("  c=8  : 3.6-5.1 s/item measured  ->  10.2-14.6 h")
print("  c=16 : 4.0 s/item measured      ->      11.3 h")
print("  c=24 : 8.4 s/item measured      ->      23.9 h  (slower, the endpoint throttles)")
