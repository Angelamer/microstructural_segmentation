import os
import re
import shutil
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm   # progress bar
from PIL import Image   # for conversion

_YX_SUFFIX_RE = re.compile(r"(\d+)[_\s]+(\d+)\.(?:tif|tiff)$", re.IGNORECASE)

def _extract_yx_from_name(name: str) -> Tuple[int, int]:
    """
    Extract (y, x) from a filename that ends with '<int_y>_<int_x>.tif[f]'.
    Matches the *last* two integers before the extension.
    """
    m = _YX_SUFFIX_RE.search(name.strip())
    if not m:
        raise ValueError(f"Cannot parse y,x from filename: {name}")
    y, x = int(m.group(1)), int(m.group(2))
    return y, x

def select_and_copy_tiffs(
    src_dir: os.PathLike,
    dst_dir: os.PathLike,
    x_range: Tuple[int, int] = (0, 494),
    y_range: Tuple[int, int] = (115, 380),
    overwrite: bool = False,
    out_format: str = "PNG",  # or "JPEG"
) -> List[Path]:
    """
    Select all .tif/.tiff images from `src_dir` whose filename encodes coordinates
    as '{y}_{x}.tiff' and copy those within the inclusive ranges to `dst_dir`.

    Args:
        src_dir: folder containing the .tif/.tiff images (named '{y}_{x}.tiff').
        dst_dir: folder to create and store the selected images.
        x_range: inclusive (xmin, xmax) for x.
        y_range: inclusive (ymin, ymax) for y.
        overwrite: if True, overwrite files in `dst_dir` when name collisions occur.

    Returns:
        A list of destination Paths that were copied.
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    xmin, xmax = x_range
    ymin, ymax = y_range

    # Collect all files first
    all_files = [p for p in src.iterdir() if p.suffix.lower() in {".tif", ".tiff"}]

    selected: List[Path] = []
    for p in tqdm(all_files, desc="Processing TIFFs", unit="file"):
        try:
            y, x = _extract_yx_from_name(p.name)
        except ValueError:
            continue

        if (xmin <= x <= xmax) and (ymin <= y <= ymax):
            y_new = y - ymin  # reindex y
            new_name = f"{y_new}_{x}.{out_format.lower()}"
            out_path = dst / new_name

            if out_path.exists() and not overwrite:
                continue

            try:
                # Open TIFF and convert
                with Image.open(p) as im:
                    # convert to 8-bit grayscale if needed
                    if im.mode not in ("L", "RGB"):
                        im = im.convert("L")
                    im.save(out_path, out_format)
                selected.append(out_path)
            except Exception as e:
                print(f"[warn] Skipping {p} (error: {e})")

    print(
        f"\nConverted {len(selected)} files with y reindexed: "
        f"y in [{ymin},{ymax}] → [0,{ymax-ymin}]"
    )
    return selected

if __name__ == "__main__":
    src = "/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/Processed_Images/ProcessedImages/"
    dst = "/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/Images_Valid/"
    select_and_copy_tiffs(
    src, dst, x_range=(0,494), y_range=(115,380), overwrite=False, out_format="JPEG"
)



