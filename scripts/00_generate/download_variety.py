"""
Download Real-IAD Variety from Hugging Face into reasoning_traces_gen/data/realiad-variety.

The HF repo (Real-IAD/Real-IAD_Variety) stores one zip per category under
realiadvariety_raw/<category>.zip plus a small realiadvariety_jsons.zip with the
per-category split metadata.

Per category the script: downloads the zip -> extracts to data/realiad-variety/<category>/
-> (optionally) prunes everything except the requested camera view(s) to save space
-> deletes the zip. It is resumable: categories already extracted are skipped.

Token is read from the HF_TOKEN environment variable (or `huggingface-cli login`).

WARNING on size (measured from the HF central directories):
  - Full dataset (all 5 views)      ~262 GB
  - C1 view only (top-down, 16 MP)  ~105 GB   (C1 is ~40% of the bytes)
Use --views C1 (default) to keep only what the C1-angle pipeline needs, and
process one category at a time so peak transient footprint is a single zip
(up to ~7.4 GB) rather than the whole dataset.

Usage:
  # metadata only (2.3 MB) -- needed by sample_variety.py
  python download_variety.py --jsons-only

  # a few categories, C1 view only, delete zips after extract (default)
  python download_variety.py --categories audio_jack_socket toy_tire

  # everything, C1 view only
  python download_variety.py --all

  # everything, ALL five views (262 GB -- be sure you have the space)
  python download_variety.py --all --views all --keep-zip
"""
import argparse
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_TOKEN  = os.environ.get("HF_TOKEN", "")  # set via `export HF_TOKEN=...` or `huggingface-cli login`
REPO_ID   = "Real-IAD/Real-IAD_Variety"
REPO_TYPE = "dataset"
RAW_DIR   = "realiadvariety_raw"        # folder of per-category zips in the repo
JSONS_ZIP = "realiadvariety_jsons.zip"  # per-category split metadata

DEST_ROOT = Path(__file__).parent / "data" / "realiad-variety"
# transient download area (zips deleted after use). Overridable per-process via
# VARIETY_ZIP_TMP so multiple sharded downloads don't race on a shared _zips dir.
ZIP_TMP   = Path(os.environ.get("VARIETY_ZIP_TMP", str(DEST_ROOT / "_zips")))
JSON_DIR  = DEST_ROOT / "Real-IAD_Variety_jsons"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_categories() -> list:
    """All category names (zip stems) available in the repo's raw folder."""
    api = HfApi(token=HF_TOKEN)
    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    cats = sorted(
        Path(f).stem
        for f in files
        if f.startswith(f"{RAW_DIR}/") and f.endswith(".zip")
    )
    return cats


def _find_data_root(staging: Path) -> Path:
    """Locate the directory that actually holds OK/ and NG/ inside a freshly
    extracted zip (some zips wrap content in a top-level <category>/ folder)."""
    if (staging / "OK").exists() or (staging / "NG").exists():
        return staging
    subdirs = [p for p in staging.iterdir() if p.is_dir()]
    for sd in subdirs:
        if (sd / "OK").exists() or (sd / "NG").exists():
            return sd
    # fall back to the single child if there is exactly one
    if len(subdirs) == 1:
        return subdirs[0]
    return staging


def _prune_views(cat_root: Path, allowed: set) -> int:
    """Delete png files whose camera tag (_C#_) is not in `allowed`.
    Returns count of removed files. Masks share the image's _C#_ tag so they
    are kept/removed together."""
    removed = 0
    for f in list(cat_root.rglob("*.png")):
        m = re.search(r"_C(\d)_", f.name)
        view = f"C{m.group(1)}" if m else None
        if view is None or view not in allowed:
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    # remove now-empty directories (deepest first)
    for d in sorted((p for p in cat_root.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts), reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass
    return removed


def download_jsons():
    """Fetch + extract the small split-metadata zip (needed by the sampler)."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {JSONS_ZIP} ...")
    local = hf_hub_download(
        REPO_ID, JSONS_ZIP, repo_type=REPO_TYPE, token=HF_TOKEN,
        local_dir=str(ZIP_TMP),
    )
    with zipfile.ZipFile(local) as z:
        staging = ZIP_TMP / "_jsons_staging"
        if staging.exists():
            shutil.rmtree(staging)
        z.extractall(staging)
    # the zip contains a Real-IAD_Variety_jsons/ folder of 160 json files
    root = _find_jsons_root(staging)
    for jf in root.glob("*.json"):
        shutil.copy2(jf, JSON_DIR / jf.name)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        Path(local).unlink()
    except OSError:
        pass
    n = len(list(JSON_DIR.glob("*.json")))
    print(f"  -> {JSON_DIR}  ({n} category json files)")


def _find_jsons_root(staging: Path) -> Path:
    direct = list(staging.glob("*.json"))
    if direct:
        return staging
    for sd in staging.rglob("*"):
        if sd.is_dir() and list(sd.glob("*.json")):
            return sd
    return staging


def download_category(cat: str, allowed_views, keep_zip: bool, force: bool):
    final = DEST_ROOT / cat
    if final.exists() and not force:
        print(f"[skip] {cat} (already extracted)")
        return
    if final.exists() and force:
        shutil.rmtree(final)

    print(f"[get ] {cat} ...")
    local = hf_hub_download(
        REPO_ID, f"{RAW_DIR}/{cat}.zip", repo_type=REPO_TYPE, token=HF_TOKEN,
        local_dir=str(ZIP_TMP),
    )
    local = Path(local)

    staging = DEST_ROOT / f".staging_{cat}"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(local) as z:
        z.extractall(staging)

    root = _find_data_root(staging)
    shutil.move(str(root), str(final))
    shutil.rmtree(staging, ignore_errors=True)

    if allowed_views != "all":
        removed = _prune_views(final, set(allowed_views))
        print(f"       pruned {removed} non-{','.join(sorted(allowed_views))} pngs")

    if not keep_zip:
        try:
            local.unlink()
        except OSError:
            pass

    kept = len(list(final.rglob("*.png")))
    print(f"[done] {cat}  ({kept} png files kept)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_views(s: str):
    if s.strip().lower() == "all":
        return "all"
    views = {v.strip().upper() for v in s.split(",") if v.strip()}
    bad = [v for v in views if not re.fullmatch(r"C[1-5]", v)]
    if bad:
        sys.exit(f"Invalid view(s): {bad}. Use C1..C5 or 'all'.")
    return views


def main():
    ap = argparse.ArgumentParser(description="Download Real-IAD Variety into data/realiad-variety.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Download every category")
    g.add_argument("--categories", nargs="+", help="Specific category names to download")
    ap.add_argument("--jsons-only", action="store_true", help="Only fetch the 2.3 MB split metadata, no images")
    ap.add_argument("--views", default="C1", help="Camera view(s) to keep: 'C1' (default), 'C1,C3', or 'all'")
    ap.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip(s) instead of deleting after extract")
    ap.add_argument("--force", action="store_true", help="Re-download/extract even if the category folder exists")
    ap.add_argument("--list", action="store_true", help="List available categories and exit")
    args = ap.parse_args()

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    ZIP_TMP.mkdir(parents=True, exist_ok=True)

    if args.list:
        for c in list_categories():
            print(c)
        return

    # always make sure the split metadata is present (cheap, sampler needs it)
    if not list(JSON_DIR.glob("*.json")):
        download_jsons()
    if args.jsons_only:
        return

    allowed = parse_views(args.views)

    if args.all:
        cats = list_categories()
    elif args.categories:
        cats = args.categories
    else:
        ap.error("Specify --all, --categories <names>, or --jsons-only")

    print(f"\n{len(cats)} categor(y/ies) | views={args.views} | keep_zip={args.keep_zip}\n")
    for i, c in enumerate(cats, 1):
        print(f"({i}/{len(cats)})", end=" ")
        try:
            download_category(c, allowed, args.keep_zip, args.force)
        except Exception as e:
            print(f"[ERROR] {c}: {e}")

    # tidy transient dir if empty
    try:
        shutil.rmtree(ZIP_TMP)
    except OSError:
        pass
    print("\nDone.")


if __name__ == "__main__":
    main()
