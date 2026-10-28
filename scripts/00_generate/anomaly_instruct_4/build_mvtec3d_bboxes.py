"""Create MVTec3D bbox prompt images under IAD-VLMs; source data stay untouched."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from tqdm import tqdm

import config
from dataset_utils import (
    deduplicated_records,
    existing_bbox_path,
    image_paths,
    mask_bbox,
    mask_paths,
)


def draw_bbox_prompt(
    query_path: Path, mask_path: Path, output_path: Path, width: int = 4
) -> list[int] | None:
    bbox, _ = mask_bbox(mask_path, allow_empty=True)
    with Image.open(query_path) as image:
        rendered = image.convert("RGB")
    if bbox is not None:
        draw = ImageDraw.Draw(rendered)
        draw.rectangle(tuple(bbox), outline=(255, 0, 0), width=width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    return bbox


def build(*, overwrite: bool = False, max_samples: int = 0, line_width: int = 4) -> tuple[int, int]:
    records = [
        record
        for _, record, _ in deduplicated_records("mvtec3d")
        if record.get("metadata", {}).get("anomaly") is True
    ]
    if max_samples:
        records = records[:max_samples]

    written = skipped = 0
    for record in tqdm(records, desc="MVTec3D bbox prompts"):
        queries = image_paths(record, "mvtec3d")
        masks = mask_paths(record, "mvtec3d")
        if len(queries) != len(masks):
            raise ValueError(f"query/mask count mismatch for {record.get('id')}")
        mask_by_name = {path.name: path for path in masks}
        for query in queries:
            mask = mask_by_name.get(query.name)
            if mask is None:
                raise ValueError(f"No matching mask for {query}")
            output = existing_bbox_path(query, "mvtec3d")
            if output.exists() and not overwrite:
                skipped += 1
                continue
            draw_bbox_prompt(query, mask, output, line_width)
            written += 1
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw red MVTec3D bbox prompts into the IAD-VLMs work directory."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="3D objects; 0 means all")
    parser.add_argument("--line-width", type=int, default=4)
    args = parser.parse_args()
    if not config.SOURCE_ROOT.exists():
        parser.error(f"source root not found: {config.SOURCE_ROOT}")
    written, skipped = build(
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        line_width=args.line_width,
    )
    print(f"written={written} skipped_existing={skipped}")
    print(f"output_root={config.DERIVED_BBOX_DIR / 'mvtec3d'}")


if __name__ == "__main__":
    main()
