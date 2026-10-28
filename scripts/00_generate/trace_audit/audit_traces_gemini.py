#!/usr/bin/env python3
"""Step 4 of the trace grounding audit: run the grounding prompt over the corpus.

Request code is the one from papers_reward_study/gemini_vision.py, extended with
retries, sharding, resume and an error file. Model gemini-3.7-flash, thinking level
low, temperature 0, at most 3 images per call (the API limit here is 8).

Images per item, as designed:
  anomalous with a mask : QUERY, OVERLAY (red mask), REFERENCE NORMAL
  anomalous, no mask    : QUERY, REFERENCE NORMAL          (556 items have no C1 mask)
  normal                : QUERY only

The overlay is built with the same recipe as create_overlay() in
reasoning_traces_gen_laptop_adnane/verify_traces_openai.py: mask resized to the image
with NEAREST, every mask pixel above 127 painted (255, 0, 0) at alpha 128, then
alpha_composite. Here the per-pixel loop is replaced by Image.point plus putalpha,
which produces the identical result and runs about a thousand times faster.

The key is read from GEMINI_API_KEY and is never written to disk.

Examples
  GEMINI_API_KEY=... python3 audit_traces_gemini.py --limit 10 --out smoke.jsonl
  GEMINI_API_KEY=... python3 audit_traces_gemini.py --shard 0/4 --concurrency 8
"""
import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.jsonl"
PROMPT = HERE / "prompt_grounding.txt"
MODEL = "gemini-3.7-flash"
URL_TMPL = ("https://aiplatform.googleapis.com/v1/publishers/google/models/"
            "{model}:streamGenerateContent?key={key}")


# --------------------------------------------------------------------- images
def b64_path(path):
    with open(path, "rb") as f:
        data = f.read()
    mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
    return base64.b64encode(data).decode(), mime


def b64_img(img, quality=85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def make_overlay(image_path, mask_path, color=(255, 0, 0), alpha=128):
    """Same recipe as verify_traces_openai.create_overlay()."""
    base = Image.open(image_path).convert("RGBA")
    mask = Image.open(mask_path).convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.NEAREST)
    a = mask.point(lambda v: alpha if v > 127 else 0)
    ov = Image.new("RGBA", base.size, (*color, 0))
    ov.putalpha(a)
    return Image.alpha_composite(base, ov).convert("RGB")


# --------------------------------------------------------------------- request
def build_parts(row, system_prompt):
    parts = [{"text": system_prompt}]
    roles = []

    parts.append({"text": "\n[IMAGE 1 of the set: QUERY image, the sample under inspection]"})
    d, m = b64_path(row["image_path"])
    parts.append({"inlineData": {"mimeType": m, "data": d}})
    roles.append("query")

    if row["gold_label"] == "yes":
        if row["mask_path"]:
            parts.append({"text": "[NEXT IMAGE: OVERLAY, the same query image with the "
                                  "ground-truth defect region filled in red]"})
            d2, m2 = b64_img(make_overlay(row["image_path"], row["mask_path"]))
            parts.append({"inlineData": {"mimeType": m2, "data": d2}})
            roles.append("overlay")
        else:
            parts.append({"text": "[NO OVERLAY: this sample is anomalous but the defect is not "
                                  "visible in this camera view, so no mask exists for it]"})
        parts.append({"text": "[NEXT IMAGE: REFERENCE NORMAL, a defect-free unit of the same "
                              "product from the same camera. Context only, not the sample "
                              "under inspection]"})
        d3, m3 = b64_path(row["ref_normal_path"])
        parts.append({"inlineData": {"mimeType": m3, "data": d3}})
        roles.append("reference_normal")
    else:
        parts.append({"text": "[NO FURTHER IMAGES: this sample is labelled normal, "
                              "there is no defect mask]"})

    meta = [f"PRODUCT: {row['product']}",
            f"GOLD VERDICT (from the dataset label, not negotiable): "
            f"{'Yes' if row['gold_label'] == 'yes' else 'No'}"]
    if row["gold_label"] == "yes":
        if row.get("gold_type"):
            meta.append(f"GROUND-TRUTH DEFECT TYPE: {row['gold_type']}")
        if row.get("gold_location"):
            meta.append(f"GROUND-TRUTH DEFECT REGION (from the mask): {row['gold_location']}")
    parts.append({"text": "\n" + "\n".join(meta)})
    parts.append({"text": f"\nTRACE TEXT TO AUDIT:\n{row['trace']}"})
    return parts, roles


def post(url, parts, max_tokens, timeout):
    body = {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.0,
                                 "maxOutputTokens": max_tokens,
                                 "responseMimeType": "application/json",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        chunks = json.load(r)
    out, finish, usage = [], None, {}
    for ch in chunks:
        u = ch.get("usageMetadata") or {}
        if "totalTokenCount" in u:          # only the final chunk carries the real counts
            usage = u
        for cand in ch.get("candidates", []):
            finish = cand.get("finishReason") or finish
            for p in cand.get("content", {}).get("parts", []):
                if "text" in p:
                    out.append(p["text"])
    return "".join(out), finish, usage


class Retryable(Exception):
    pass


def call_with_retries(url, parts, args):
    """Exponential backoff on 429 and 5xx and on transport errors. Hard cap on tries."""
    delay = args.backoff_base
    last = None
    max_tokens = args.max_tokens
    for attempt in range(1, args.max_retries + 1):
        try:
            txt, finish, usage = post(url, parts, max_tokens, args.timeout)
            if finish == "MAX_TOKENS":
                max_tokens = min(int(max_tokens * 1.5), 32000)
                raise Retryable(f"truncated (finishReason=MAX_TOKENS), raising cap to {max_tokens}")
            # A cut-off stream still parses as a valid JSON array, but the last chunk
            # carries neither a finishReason nor the usage counters. Without this guard
            # the truncated text reaches parse_verdict and dies there as a parse_error.
            # It cost 12 of the first 200 calibration items.
            if finish is None or "totalTokenCount" not in usage:
                raise Retryable("incomplete stream (no finishReason / no usage), retrying")
            return txt, finish, attempt, usage
        except urllib.error.HTTPError as e:
            code = e.code
            detail = ""
            try:
                detail = e.read().decode()[:300]
            except Exception:
                pass
            last = f"HTTP {code} {detail}"
            if code != 429 and code < 500:
                raise RuntimeError(last)          # 4xx other than 429 will not fix itself
            if code == 429:
                # RESOURCE_EXHAUSTED is a shared-quota wall, not a transient blip. It
                # cost 23 of 50 items in the concurrency-8 calibration probe. Back off
                # from a much higher floor than the 5xx path uses.
                delay = max(delay, 30.0)
        except Retryable as e:
            last = str(e)
        except Exception as e:                     # URLError, timeout, malformed stream
            last = f"{type(e).__name__}: {e}"
        if attempt < args.max_retries:
            time.sleep(min(delay, args.backoff_max) * (1.0 + 0.25 * random.random()))
            delay *= 2
    raise RuntimeError(f"gave up after {args.max_retries} tries: {last}")


# --------------------------------------------------------------------- parsing
def parse_verdict(txt):
    t = txt.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        j = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if not m:
            raise
        j = json.loads(m.group(0))
    if not isinstance(j, dict):
        raise ValueError("model did not return a JSON object")
    cat = j.get("category")
    if cat not in ("completely_correct", "correct", "wrong", "completely_wrong"):
        raise ValueError(f"bad category {cat!r}")
    issues = j.get("issues") or []
    if not isinstance(issues, list):
        issues = [{"claim": "", "why_wrong": str(issues), "what_it_should_be": ""}]
    clean = []
    for it in issues:
        if isinstance(it, dict):
            clean.append({"issue_type": str(it.get("issue_type", "") or ""),
                          "claim": str(it.get("claim", "")),
                          "why_wrong": str(it.get("why_wrong", "")),
                          "what_it_should_be": str(it.get("what_it_should_be", ""))})
        else:
            clean.append({"issue_type": "", "claim": "", "why_wrong": str(it),
                          "what_it_should_be": ""})
    # prompt v2 only: the judge writes down every count it made before it decides
    counts = j.get("counts") or []
    if not isinstance(counts, list):
        counts = []
    cclean = []
    for c in counts:
        if not isinstance(c, dict):
            continue

        def _n(v):
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
        cclean.append({"what": str(c.get("what", "")),
                       "trace_says": _n(c.get("trace_says")),
                       "i_counted": _n(c.get("i_counted")),
                       "sure": bool(c.get("sure", True))})
    try:
        conf = float(j.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    return {"category": cat,
            "confidence": max(0.0, min(1.0, conf)),
            "not_visible_in_view": bool(j.get("not_visible_in_view", False)),
            "counts": cclean,
            "issues": clean,
            "tags_ok": bool(j.get("tags_ok", True)),
            "corrected_trace": str(j.get("corrected_trace") or "")}


# --------------------------------------------------------------------- driver
class Writer:
    """One JSON object per line, flushed and fsynced so a kill never loses a record."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, obj):
        with self.lock:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--prompt", default=str(PROMPT))
    ap.add_argument("--out", default=str(HERE / "audit_grounding.jsonl"))
    ap.add_argument("--errors", default=None, help="default: <out>.errors.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="stop after N items, 0 = all")
    ap.add_argument("--shard", default="0/1", help="i/n, take items where index %% n == i")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--backoff-base", type=float, default=4.0)
    ap.add_argument("--backoff-max", type=float, default=300.0)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--retry-errored", action="store_true",
                    help="also redo items whose stored record has status != ok")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--total", type=int, default=0,
                    help="size of the whole sweep, so a sharded run can report global "
                         "progress against the full corpus. 0 = this shard only")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    url = URL_TMPL.format(model=args.model, key=key)

    errors_path = args.errors or (args.out + ".errors.jsonl")
    system_prompt = open(args.prompt).read()
    prompt_sha1 = hashlib.sha1(system_prompt.encode()).hexdigest()[:12]

    rows = [json.loads(l) for l in open(args.corpus)]
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        sys.exit(f"bad --shard {args.shard}")
    rows = [r for k, r in enumerate(rows) if k % n == i]

    done = load_done(args.out)
    if args.retry_errored and os.path.exists(args.out):
        keep, redo = [], set()
        with open(args.out) as f:
            for line in f:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if o.get("status") == "ok":
                    keep.append(line)
                else:
                    redo.add(o["id"])
        if redo:
            with open(args.out, "w") as f:
                f.writelines(keep)
            done -= redo
            print(f"[resume] dropped {len(redo)} errored records, they will be redone")

    todo = [r for r in rows if r["id"] not in done]
    already = len(rows) - len(todo)
    if args.limit:
        todo = todo[:args.limit]

    print(f"[init] model={args.model} prompt_sha1={prompt_sha1} shard={args.shard} "
          f"concurrency={args.concurrency}")
    print(f"[init] shard items {len(rows)} | already done {already} | to do now {len(todo)}")
    print(f"[init] out={args.out}")
    print(f"[init] errors={errors_path}")
    if not todo:
        print("[done] nothing to do")
        return 0

    out_w, err_w = Writer(args.out), Writer(errors_path)
    counters = {"ok": 0, "parse_error": 0, "api_error": 0}
    lock = threading.Lock()
    t0 = time.time()
    # Global bookkeeping for the progress line. All shards append to one out file,
    # so len(done) is how much of the whole sweep is already on disk.
    start_done = len(done)
    g_total = args.total or (already + len(todo))

    def work(row):
        t1 = time.time()
        rec = {"id": row["id"], "split": row["split"], "product": row["product"],
               "gold_label": row["gold_label"], "gold_type": row["gold_type"],
               "gold_location": row["gold_location"],
               "image_path": row["image_path"], "mask_path": row["mask_path"],
               "ref_normal_path": row["ref_normal_path"],
               "model": args.model, "prompt_sha1": prompt_sha1}
        raw, attempts = "", 0
        try:
            parts, roles = build_parts(row, system_prompt)
            rec["images_sent"] = roles
            raw, finish, attempts, usage = call_with_retries(url, parts, args)
            rec["attempts"] = attempts
            rec["finish_reason"] = finish
            rec["prompt_tokens"] = usage.get("promptTokenCount")
            rec["output_tokens"] = usage.get("candidatesTokenCount")
            rec["thought_tokens"] = usage.get("thoughtsTokenCount")
            rec["total_tokens"] = usage.get("totalTokenCount")
            rec.update(parse_verdict(raw))
            rec["status"] = "ok"
        except json.JSONDecodeError as e:
            rec.update({"status": "parse_error", "error": f"JSONDecodeError: {e}",
                        "attempts": attempts, "raw": raw[:4000],
                        "category": None, "confidence": 0.0, "issues": [],
                        "counts": [], "not_visible_in_view": None,
                        "tags_ok": None, "corrected_trace": ""})
        except Exception as e:
            status = "parse_error" if raw else "api_error"
            rec.update({"status": status, "error": f"{type(e).__name__}: {e}",
                        "attempts": attempts, "raw": raw[:4000],
                        "category": None, "confidence": 0.0, "issues": [],
                        "counts": [], "not_visible_in_view": None,
                        "tags_ok": None, "corrected_trace": ""})
        rec["latency_s"] = round(time.time() - t1, 2)
        out_w.write(rec)
        if rec["status"] != "ok":
            err_w.write({"id": rec["id"], "status": rec["status"],
                         "error": rec.get("error"), "attempts": rec.get("attempts"),
                         "raw": rec.get("raw", "")[:4000], "ts": time.time()})
        with lock:
            counters[rec["status"]] += 1
            n_done = sum(counters.values())
            if n_done % args.log_every == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / el if el else 0.0
                g_done = start_done + n_done
                g_left = max(g_total - g_done, 0)
                eta_s = g_left / rate if rate else 0.0
                fin = time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.localtime(time.time() + eta_s))
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"done {g_done}/{g_total} | left {g_left} "
                      f"| shard {args.shard} {n_done}/{len(todo)} "
                      f"| rate {rate:.3f} it/s ({rate * 3600:.0f} it/h) "
                      f"| elapsed {el / 60:.1f} min | eta {eta_s / 3600:.2f} h "
                      f"| finish ~{fin} "
                      f"| ok {counters['ok']} parse_error {counters['parse_error']} "
                      f"api_error {counters['api_error']}", flush=True)
        return rec["status"]

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, todo))

    el = time.time() - t0
    print(f"[done] {sum(counters.values())} items in {el/60:.1f} min | {counters}")
    return 1 if counters["api_error"] or counters["parse_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
