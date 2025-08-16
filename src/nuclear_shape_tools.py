# This script is designed to be paired with the nuclear_shape_analysis.py and main_analysis.py scritps
# Here lie the tools for loading and cleaning data from CellProfiler csv. files

from __future__ import annotations
from typing import Dict, List, Optional, Iterable
from pandas.api.types import is_numeric_dtype

import csv
import numpy as np
import os
import pandas as pd
import re

# -------------------- I/O --------------------


def load_cp_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# -------------------- ID / prefix / merge --------------------

# Canonical cell and treatment names
CELL_CANON = {"hela": "HeLa", "rpe": "RPE", "nih3t3": "NIH3T3"}
TX_CANON = {"ctl": "ctl", "ble": "ble", "h11": "h11", "y27": "y27"}

# new schema: "CellType_Treatment_*"
NEW_SCHEMA_RX = re.compile(r"^\s*(hela|rpe|nih3t3)[-_](ctl|ble|h11|y27)\b", re.I)

# fallbacks for old schema (search anywhere in filename/path)
CELL_FALLBACK = [
    ("HeLa", re.compile(r"(?<![A-Za-z0-9])hela(?![A-Za-z0-9])", re.I)),
    ("RPE", re.compile(r"(?<![A-Za-z0-9])rpe(?![A-Za-z0-9])", re.I)),
    (
        "NIH3T3",
        re.compile(
            r"(?<![A-Za-z0-9])nih\s*3t3(?![A-Za-z0-9])|(?<![A-Za-z0-9])nih3t3(?![A-Za-z0-9])",
            re.I,
        ),
    ),
]
TX_FALLBACK = [
    ("ctl", re.compile(r"(?<![A-Za-z0-9])ctl(?![A-Za-z0-9])|control", re.I)),
    ("ble", re.compile(r"(?<![A-Za-z0-9])ble(?![A-Za-z0-9])|bleb|blebbistatin", re.I)),
    ("h11", re.compile(r"(?<![A-Za-z0-9])h11(?![A-Za-z0-9])|hesperadin", re.I)),
    ("y27", re.compile(r"(?<![A-Za-z0-9])y27(?![A-Za-z0-9])|y[-_ ]?27632", re.I)),
]


def _canon_cell(x: Optional[str]) -> str:
    if not x:
        return "unknown"
    return CELL_CANON.get(x.lower(), x)


def _canon_tx(x: Optional[str]) -> str:
    if not x:
        return "unknown"
    return TX_CANON.get(x.lower(), x.lower())


def _first_match(text: str, patterns) -> Optional[str]:
    for label, rx in patterns:
        if rx.search(text):
            return label
    return None


def _load_label_map(label_map_path: Optional[str]) -> List[Dict[str, str]]:
    """
    Optional mapping CSV with columns:
      contains, cell_type, treatment
    Each row: if 'contains' substring is found in filename/path, set labels.
    """
    if not label_map_path:
        return []
    rules: List[Dict[str, str]] = []
    with open(label_map_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("contains"):
                continue
            rules.append(
                {
                    "contains": row["contains"],
                    "cell_type": row.get("cell_type", ""),
                    "treatment": row.get("treatment", ""),
                }
            )
    return rules


def add_group_columns(
    df: pd.DataFrame,
    filename_col: str = "FileName",
    path_cols: Optional[List[str]] = None,
    default_cell_type: Optional[str] = None,
    default_treatment: Optional[str] = None,
    label_map_path: Optional[str] = None,
) -> pd.DataFrame:

    out = df.copy()

    # build a text field per row for matching
    if filename_col in out.columns:
        base = out[filename_col].astype(str)
        pcols = path_cols or [c for c in out.columns if c.startswith("PathName_")]
        cols_to_join = [filename_col] + pcols
    else:
        # Use any FileName_* and PathName_* columns
        cols_to_join = [
            c for c in out.columns if c.startswith(("FileName_", "PathName_"))
        ]

    if cols_to_join:
        joined = out[cols_to_join].astype(str).agg(" ".join, axis=1)
    else:
        joined = pd.Series([""] * len(out))

    # optional mapping rules
    rules = _load_label_map(label_map_path)

    cell_vals: List[str] = []
    tx_vals: List[str] = []

    for raw in joined.fillna(""):
        txt = str(raw)
        # 1) New schema prefix: "CellType_Tx_*"
        m = NEW_SCHEMA_RX.match(os.path.basename(txt.strip()))
        if m:
            cell = _canon_cell(m.group(1))
            tx = _canon_tx(m.group(2))
        else:
            # 2) Mapping (substring contains)
            cell = None
            tx = None
            for rule in rules:
                if rule["contains"] and rule["contains"].lower() in txt.lower():
                    if rule["cell_type"]:
                        cell = _canon_cell(rule["cell_type"])
                    if rule["treatment"]:
                        tx = _canon_tx(rule["treatment"])
                    # keep scanning; later rules might fill the other label

            # 3) Fallback regex anywhere in filename/path
            if cell is None:
                cell = _first_match(txt, CELL_FALLBACK)
            if tx is None:
                tx = _first_match(txt, TX_FALLBACK)

        # 4) CLI defaults (if still missing)
        cell = cell or default_cell_type
        tx = tx or default_treatment

        cell_vals.append(_canon_cell(cell))
        tx_vals.append(_canon_tx(tx))

    # Create columns, with consistent category order
    out["cell_type"] = pd.Categorical(
        cell_vals, categories=["HeLa", "RPE", "NIH3T3", "unknown"], ordered=False
    )
    out["treatment"] = pd.Categorical(
        tx_vals, categories=["ctl", "ble", "h11", "y27", "unknown"], ordered=True
    )
    out["group"] = pd.Categorical(
        out["cell_type"].astype(str) + "_" + out["treatment"].astype(str)
    )

    return out


# columns that are considered IDs and should not be prefixed
ID_PATTERNS = [
    r"^ImageNumber$",
    r"^ObjectNumber$",
    r"^Metadata_.*",
    r"^Parent_.*",
    r"^FileName_.*",
    r"^PathName_.*",
]


def _is_id_col(name: str) -> bool:
    return any(re.match(p, name) for p in ID_PATTERNS)


def prefix_non_id(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    rename_map: Dict[str, str] = {}
    for c in out.columns:
        if not _is_id_col(c) and not c.startswith(prefix + "_"):
            rename_map[c] = f"{prefix}_{c}"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def merge_on_ids(nuc_df: pd.DataFrame, act_df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        k
        for k in ["ImageNumber", "ObjectNumber"]
        if k in nuc_df.columns and k in act_df.columns
    ]
    if not keys:
        nuc_df = nuc_df.reset_index().rename(columns={"index": "_row"})
        act_df = act_df.reset_index().rename(columns={"index": "_row"})
        keys = ["_row"]
    return pd.merge(nuc_df, act_df, on=keys, how="inner", suffixes=("", "_actdup"))


def drop_pathnames(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("PathName_")]
    return df.drop(columns=cols, errors="ignore")


def consolidate_filenames(
    df: pd.DataFrame, prefer: str = "FileName_Act", output_col: str = "FileName"
) -> pd.DataFrame:
    df = df.copy()
    file_cols = [c for c in df.columns if c.startswith("FileName_")]
    chosen = prefer if prefer in df.columns else (file_cols[0] if file_cols else None)
    if chosen:
        df[output_col] = df[chosen].astype(str).map(os.path.basename)
    return df.drop(columns=file_cols, errors="ignore")


# -------------------- problem-file filtering --------------------


def load_problematic_files(filepath: str) -> List[str]:
    names: List[str] = []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(("%", "#")):
                continue
            names.append(os.path.basename(s))
    return names


def remove_problematic_files(
    df: pd.DataFrame, bad_list: List[str], file_column: str
) -> pd.DataFrame:
    if not bad_list or file_column not in df.columns:
        return df
    base = df[file_column].astype(str).map(os.path.basename)
    mask = ~base.isin(set(bad_list))
    return df.loc[mask].reset_index(drop=True)


# -------------------- unit conversion (px -> µm) --------------------

CONVERT_COORDINATES_DEFAULT = True

_EXCLUDE_PATTERNS = [
    r"Zernike",
    r"NormalizedMoment",
    r"Haralick",
    r"Texture",
    r"Eccentricity",
    r"Solidity",
    r"Extent",
    r"FormFactor",
    r"Compactness",
    r"Elongation",
    r"^Orientation$",
    r"\bAngle\b",
    r"^Metadata_",
    r"^Parent_",
    r"^PathName_",
    r"^FileName_",
]
_AREA_SPECIFIC = [
    r"(?:^|_)AreaShape_ConvexArea$",
    r"(?:^|_)AreaShape_BoundingBoxArea$",
    r"(?:^|_)Area(?:$|_)",  # e.g., nuc_Area
]

_LENGTH_SPECIFIC = [
    r"(?:^|_)Perimeter$",
    r"(?:^|_)FeretDiameter(?:Max|Min)?$",
    r"(?:^|_)MaxFeretDiameter$",
    r"(?:^|_)MinFeretDiameter$",
    r"(?:^|_)MajorAxisLength$",
    r"(?:^|_)MinorAxisLength$",
    r"(?:^|_)EquivalentDiameter$",
]
_COORD_SPECIFIC = [
    r"(?:^|_)AreaShape_BoundingBox(?:Minimum|Maximum)[XY]$",
    r"(?:^|_)AreaShape_BoundingBox(?:Width|Height)$",
    r"(?:^|_)Location_(?:Center|Centroid)_(?:X|Y)$",
    r"(?:^|_)CenterMass(?:X|Y)$",
]
_GENERIC_LENGTH = [
    r"(?:^|_)AreaShape_.*\b(Diameter|Length|Radius)\b",
    r"\b(Diameter|Length|Radius)\b",
]

_EXCLUDE_RX = [re.compile(p) for p in _EXCLUDE_PATTERNS]
_AREA_RX = [re.compile(p) for p in _AREA_SPECIFIC]
_LENGTH_RX = [re.compile(p) for p in _LENGTH_SPECIFIC]
_COORD_RX = [re.compile(p) for p in _COORD_SPECIFIC]
_GENLEN_RX = [re.compile(p) for p in _GENERIC_LENGTH]


def _unit_power(col: str, convert_coordinates: bool) -> Optional[int]:
    for rx in _EXCLUDE_RX:
        if rx.search(col):
            return None
    for rx in _AREA_RX:
        if rx.search(col):
            return 2
    for rx in _LENGTH_RX:
        if rx.search(col):
            return 1
    if convert_coordinates:
        for rx in _COORD_RX:
            if rx.search(col):
                return 1
    for rx in _GENLEN_RX:
        if rx.search(col):
            return 1
    return None


def convert_units(
    df: pd.DataFrame,
    pixel_size_um: Optional[float],
    convert_coordinates: bool = CONVERT_COORDINATES_DEFAULT,
) -> pd.DataFrame:
    if pixel_size_um is None or not np.isfinite(pixel_size_um):
        return df
    out = df.copy()
    for c in out.select_dtypes(include=[np.number]).columns:
        p = _unit_power(c, convert_coordinates=convert_coordinates)
        if p:
            out[c] = out[c] * (pixel_size_um**p)
    return out


# -------------------- additional metrics --------------------


def add_aspect_ratio(
    df: pd.DataFrame,
    prefixes: Iterable[str] = ("nuc", "act"),
    overwrite: bool = False,
) -> pd.DataFrame:

    out = df.copy()

    for prefix in prefixes:
        target_col = f"{prefix}_AreaShape_AspectRatio"
        if (not overwrite) and (target_col in out.columns):
            continue  # already present

        # Preferred (CellProfiler standard names)
        minor_candidates = [f"{prefix}_AreaShape_MinorAxisLength"]
        major_candidates = [f"{prefix}_AreaShape_MajorAxisLength"]

        # Fallback: any column under this prefix ending with MinorAxisLength/MajorAxisLength
        rx_minor = re.compile(rf"^{re.escape(prefix)}_.*MinorAxisLength$")
        rx_major = re.compile(rf"^{re.escape(prefix)}_.*MajorAxisLength$")

        if minor_candidates[0] not in out.columns:
            minor_candidates += [c for c in out.columns if rx_minor.match(c)]
        if major_candidates[0] not in out.columns:
            major_candidates += [c for c in out.columns if rx_major.match(c)]

        minor_col = next((c for c in minor_candidates if c in out.columns), None)
        major_col = next((c for c in major_candidates if c in out.columns), None)

        if not minor_col or not major_col:
            # Nothing to do for this prefix; silently skip
            continue

        minor = out[minor_col].astype(float)
        major = out[major_col].astype(float)

        # Avoid divide-by-zero; ratio is unitless
        ratio = np.where(major > 0, minor / major, np.nan).astype(float)
        out[target_col] = ratio

    return out


# -------------------- selection / outliers --------------------


# columns to keep
def select_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    present = [c for c in keep_cols if c in df.columns]
    return df[present].copy()


# remove outliers using IQR on area and intensity
def _log1p_series(s: pd.Series) -> pd.Series:
    """Return log1p(s) but guaranteed as a pandas Series with the same index."""
    arr = np.log1p(s.astype(float).to_numpy())
    return pd.Series(arr, index=s.index)

def _robust_z(x: pd.Series) -> pd.Series:
    """Robust z-score using MAD (median absolute deviation)."""
    s = x.astype(float)
    med = float(np.nanmedian(s))
    mad = float(np.nanmedian(np.abs(s - med)))
    if not np.isfinite(mad) or mad == 0.0:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    denom = 1.4826 * mad  # MAD->SD scaling for normal dist
    return (s - med) / denom


def _first_existing(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat)
        for c in df.columns:
            if rx.fullmatch(c) or rx.search(c):
                return c
    return None


def flag_outliers_per_group(
    df: pd.DataFrame,
    group_cols: List[str],
    cols_mad_log: List[str],
    z_cutoff: float = 3.5,
    mitotic_rule: bool = True,
    mitotic_intensity_col: Optional[str] = None,
    mitotic_area_col: Optional[str] = None,
    mitotic_int_hi: float = 2.5,
    mitotic_area_lo: float = -2.0,
) -> pd.DataFrame:
    """
    Add 'qc_keep' and 'qc_reason' columns based on robust MAD z-scores per group.
    - cols_mad_log: features to log-transform before MAD (e.g., areas, intensities)
    - mitotic_rule: optional rule: high nuclear intensity & low cell area → drop
    """
    out = df.copy()
    if "qc_keep" not in out.columns:
        out["qc_keep"] = True
    if "qc_reason" not in out.columns:
        out["qc_reason"] = ""
    
    # If nothing to do, return a copy immediately (still a DataFrame)
    if not cols_mad_log and not mitotic_rule:
        return out

    # Infer defaults for mitotic rule, if requested
    if mitotic_rule:
        if mitotic_intensity_col is None:
            mitotic_intensity_col = _first_existing(out, [r"^nuc_.*Intensity.*MeanIntensity$"])
        if mitotic_area_col is None:
            mitotic_area_col = _first_existing(out, [r"^act_.*AreaShape_Area$"])

    # Build group iterator as (group_key, index_of_rows)
    if group_cols:
        gp = out.groupby(group_cols, dropna=False, observed=False)
        group_iter = gp.groups.items()
    else:
        group_iter = [(None, out.index)]

    for _gkey, idx_vals in group_iter:
        idx = pd.Index(idx_vals)

        # Univariate robust MAD on log1p for each requested column
        for col in cols_mad_log:
            if col not in out.columns or not is_numeric_dtype(out[col]):
                continue
            series = out.loc[idx, col].astype(float)
            z = _robust_z(_log1p_series(series))
            mask = (z.abs() > z_cutoff).fillna(False)
            if mask.any():
                rows = mask.index[mask.to_numpy()]  # indices to set
                out.loc[rows, "qc_keep"] = False
                prev = out.loc[rows, "qc_reason"].astype(str)
                reason = f"mad:{col}"
                out.loc[rows, "qc_reason"] = np.where(
                    prev.str.len() > 0, prev + "|" + reason, reason
                )

        # Mitotic-like rule: high nuc intensity & low cell area
        if mitotic_rule and mitotic_intensity_col and mitotic_area_col:
            if mitotic_intensity_col in out.columns and mitotic_area_col in out.columns:
                s_int = out.loc[idx, mitotic_intensity_col].astype(float)
                s_area = out.loc[idx, mitotic_area_col].astype(float)
                z_int = _robust_z(_log1p_series(s_int))
                z_area = _robust_z(_log1p_series(s_area))
                mask_m = ((z_int > mitotic_int_hi) & (z_area < mitotic_area_lo)).fillna(False)
                if mask_m.any():
                    rows = mask_m.index[mask_m.to_numpy()]
                    out.loc[rows, "qc_keep"] = False
                    prev = out.loc[rows, "qc_reason"].astype(str)
                    reason = "mitotic_like"
                    out.loc[rows, "qc_reason"] = np.where(
                        prev.str.len() > 0, prev + "|" + reason, reason
                    )

    # <-- Always return a DataFrame
    return out

def summarize_qc(
    out: pd.DataFrame, group_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Return counts kept/removed per group with percentage."""
    if group_cols:
        gp = out.groupby(group_cols, dropna=False, observed=False)["qc_keep"]
        summary = gp.value_counts(dropna=False).unstack(fill_value=0)
    else:
        summary = out["qc_keep"].value_counts(dropna=False).to_frame().T
    # Ensure both columns exist
    for col in [True, False]:
        if col not in summary.columns:
            summary[col] = 0
    summary = summary.rename(columns={True: "kept", False: "removed"})
    summary["total"] = summary["kept"] + summary["removed"]
    summary["removed_pct"] = (summary["removed"] / summary["total"] * 100.0).round(2)
    return summary.reset_index()
