import os
import pandas as pd
import numpy as np
from pathlib import Path

def _read_tsv_matrix(path: Path, height=384, width=512):
    """
    Robustly read a TSV as a numeric (height x width) array:
    - tolerates trailing tabs / ragged rows
    - converts empty strings to NaN
    - supports ',' decimal (auto-detected)
    - clips or pads to the requested shape
    """
    # Quick sniff to see if decimals use commas
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f, "") for _ in range(5)])
    use_comma = ("," in head) and not ("\t," in head)  # crude but effective

    # Read with pandas (python engine tolerates ragged rows, fills NaN)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        engine="python",
        dtype=float,                # let pandas parse to float
        na_values=["", "NA", "NaN", "nan", " "],
        skip_blank_lines=True,
        decimal="," if use_comma else ".",
    )

    # Ensure 2D shape: pad/clip columns to width
    n_rows, n_cols = df.shape
    if n_cols < width:
        for j in range(n_cols, width):
            df[j] = np.nan
    elif n_cols > width:
        df = df.iloc[:, :width]

    # Pad/clip rows to height
    if n_rows < height:
        pad = pd.DataFrame(np.nan, index=range(height - n_rows), columns=df.columns)
        df = pd.concat([df, pad], ignore_index=True)
    elif n_rows > height:
        df = df.iloc[:height, :]

    arr = df.to_numpy(dtype=float, copy=False)
    assert arr.shape == (height, width), f"Got {arr.shape}, expected {(height, width)}"
    return arr

def _element_name_from_filename(p: Path) -> str:
    """
    Extract a concise element name from filenames like:
    'Fe Massen %.tsv', 'Al Massen%.tsv', 'O (Massen %).tsv', etc.
    Falls back to the first token before a space.
    """
    stem = p.stem.strip()
    # Keep only the first token of letters/numbers
    for token in stem.replace("(", " ").replace(")", " ").split():
        if token.isalpha() or token.isalnum():
            return token
    return stem.split()[0] if stem.split() else stem

def read_element_maps(folder, height=384, width=512):
    """
    Read all .tsv element maps in `folder` and return a long DataFrame with
    columns: x, y, <element1>, <element2>, ...; sorted by y then x.

    Each .tsv must represent a (height x width) map; ragged lines/trailing tabs are ok.
    """
    folder = Path(folder)
    tsv_files = sorted(folder.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in {folder}")

    # Base grid (x varies fastest so sorting by y, then x yields (0,0),(1,0),...)
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    df_out = pd.DataFrame({"x": xs.ravel(), "y": ys.ravel()})

    for f in tsv_files:
        elem = _element_name_from_filename(f)
        arr = _read_tsv_matrix(f, height=height, width=width)
        df_out[elem] = arr.ravel()

    # Sort by y then x
    df_out = df_out.sort_values(["y", "x"], kind="mergesort").reset_index(drop=True)
    return df_out


if __name__ == "__main__":
    folder = "/home/users/zhangqn8/storage/Partially reduced oxides 20 minutes Arbeitsbereich 3 Elementverteilungsdaten 5/EDS"  
    df = read_element_maps(folder)
    print(df.head())
    print(df.shape)
    df.to_csv("20min_element_maps.csv", index=False)
