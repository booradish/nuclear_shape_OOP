# Nuclear Shape Analysis script
#
#  This script is designed to be paired with the nuclear_shape_tool.py and main_analysis.py scritps
# Here lie the functions for running statistics, and regressional analyses on data from CellProfiler csv. files

"""

This version:
- Loads two CellProfiler CSVs (nuclei + actin/cell),
- Merges them on common ID columns (prefers ImageNumber+ObjectNumber),
- Adds a simple 'group' column if missing,
- Supports optional removal of "problem" files,
- Provides simple stats, a toy regression, and a toy "random forest" proxy,
- Exposes attributes used by the main script: df_clean, df_trimmed, and results.

If `nuclear_shape_tools.py` exists, its functions are used. Otherwise, safe
fallbacks defined here are used so you can run end-to-end right away.
"""

from __future__ import annotations
from numpy.typing import NDArray
from scipy import stats
from typing import List, Optional, Dict, Any
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pandas as pd
import seaborn as sns


# -------------------import tools from nuclear_shape_tools.py-------------------

from src.nuclear_shape_tools import (
    prefix_non_id as _prefix_non_id,
    merge_on_ids as _merge_on_ids,
    drop_pathnames as _drop_pathnames,
    consolidate_filenames as _consolidate_filenames,
    convert_units,
    add_aspect_ratio,
    add_group_columns,
    flag_outliers_per_group,
    summarize_qc,
    add_normalized_columns,
)

# -------------------define the main analysis class-------------------


# helper functions
def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce to numeric Series (non-numeric -> NaN)."""
    return pd.to_numeric(s, errors="coerce")


def _as_numeric_series(x) -> pd.Series:
    """Return x as a numeric pandas Series (non-numeric -> NaN)."""
    s = x if isinstance(x, pd.Series) else pd.Series(x)
    return pd.to_numeric(s, errors="coerce")


# MAIN ANALYSIS CLASS
class Nuclear_Shape_Analysis:
    def __init__(
        self,
        nuc_path: str,
        act_path: str,
        keep_columns: Optional[List[str]] = None,
        problem_files_path: Optional[str] = None,
        file_column: str = "FileName_Act",
        pixel_size_um: Optional[float] = None,
        convert_coordinates: bool = True,
        qc_mode: str = "robust",
        qc_drop: bool = True,
        add_aspect_ratio: bool = True,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        default_cell_type: str | None = None,
        default_treatment: str | None = None,
        label_map_path: str | None = None,
        normalize_mode: str = "none",  # "none", "zscore", "robust"
        normalize_by_group: bool = True,  # normalize within cell_type×treatment
        use_norm_for_stats: bool = False,  # whether run_stats uses normalized copies
        norm_suffix: str = "_z",
    ):
        # ----------------- paths/config -----------------
        self.nuc_path = nuc_path
        self.act_path = act_path
        self.keep_columns = keep_columns or [
            "ImageNumber",
            "ObjectNumber",
            "Metadata_Cellline",
            "Metadata_Replicate",
            "Metadata_Treatment",
            "FileName_Nuc",
        ]

        self.convert_coordinates = convert_coordinates
        self.qc_mode = qc_mode
        self.qc_drop = qc_drop
        self.add_aspect_ratio_flag = add_aspect_ratio

        self.problem_files_path = problem_files_path
        self.file_column = file_column
        self.pixel_size_um = pixel_size_um
        self.convert_coordinates = convert_coordinates
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_trimmed: Optional[pd.DataFrame] = None

        self.normalize_mode = normalize_mode
        self.normalize_by_group = normalize_by_group
        self.use_norm_for_stats = use_norm_for_stats
        self.norm_suffix = norm_suffix

        self.random_state = random_state
        self.n_jobs = n_jobs

        self.default_cell_type = default_cell_type
        self.default_treatment = default_treatment
        self.label_map_path = label_map_path

        # ---- data snapshots ----
        self.nuc_raw: Optional[pd.DataFrame] = None
        self.act_raw: Optional[pd.DataFrame] = None
        self.df_merged: Optional[pd.DataFrame] = None
        self.df_flags: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_trimmed: Optional[pd.DataFrame] = None
        self.df_norm: Optional[pd.DataFrame] = None

        # ---- columns registry ----
        self.cols: dict[str, list[str]] = (
            {}
        )  # will hold {"id": [...], "nuc": [...], "act": [...]}

        # ---- QC bookkeeping ----
        self.qc_summary: Optional[pd.DataFrame] = None
        self.n_rows_before: int = 0
        self.n_rows_after: int = 0

        # ---- results ----
        self.stats_results: Optional[Dict[str, pd.DataFrame]] = None
        self.regression_results: Optional[Dict[str, pd.DataFrame]] = None
        self.random_forest_results: Optional[pd.DataFrame] = None
        self.compare_results: Optional[pd.DataFrame] = None

    # ------------------- helpers for methods -------------------

    def df_for_analysis(self) -> pd.DataFrame:
        """One source of truth for analysis steps."""
        if self.df_trimmed is not None:
            return self.df_trimmed
        if self.df_clean is not None:
            return self.df_clean
        raise RuntimeError("No data available; run load_and_clean().")

    def _ensure_feature_lists(self) -> None:
        """Populate self.cols['id'/'nuc'/'act'] from the current analysis DF if missing."""
        if not self.cols or any(k not in self.cols for k in ("id", "nuc", "act")):
            from src.nuclear_shape_tools import trim_to_analysis

            base = self.df_for_analysis()
            trimmed, ids, nuc, act = trim_to_analysis(
                base, nuc_prefix="nuc", act_prefix="act"
            )
            # don't override user's manual trim; only set if empty
            if self.df_trimmed is None:
                self.df_trimmed = trimmed
            self.cols = {"id": ids, "nuc": nuc, "act": act}

    # -------------------analysis methods in this class-------------------

    def load_and_clean(self) -> None:
        # load
        nuc = pd.read_csv(self.nuc_path)
        act = pd.read_csv(self.act_path)
        # prefix column names
        nuc = _prefix_non_id(nuc, "nuc")
        act = _prefix_non_id(act, "act")
        # merge
        df = _merge_on_ids(nuc, act)
        # add group columns
        df = add_group_columns(
            df,
            filename_col=self.file_column,  # e.g. "FileName_Act"
            default_cell_type=self.default_cell_type,
            default_treatment=self.default_treatment,
            label_map_path=self.label_map_path,
        )

        if "group" not in df.columns:
            df["group"] = "all"

        # remove unnecessary columns
        df = _drop_pathnames(df)
        df = _consolidate_filenames(df, prefer="FileName_Nuc", output_col="FileName")

        # remove problematic files
        if self.problem_files_path and os.path.exists(self.problem_files_path):
            with open(self.problem_files_path) as f:
                bad = {
                    os.path.basename(x.strip())
                    for x in f
                    if x.strip() and not x.strip().startswith(("#", "%"))
                }
            base_col = "FileName" if "FileName" in df.columns else self.file_column
            if base_col in df.columns:
                keep_mask = ~df[base_col].astype(str).map(os.path.basename).isin(bad)
                df = df.loc[keep_mask].reset_index(drop=True)

        # unit conversion
        df = convert_units(
            df,
            pixel_size_um=self.pixel_size_um,
            convert_coordinates=self.convert_coordinates,
        )

        # add aspect ratio column for cell and nuc
        if self.add_aspect_ratio_flag:
            df = add_aspect_ratio(df, prefixes=("nuc", "act"), overwrite=False)

        # per-group robust outlier flags
        if self.qc_mode == "robust":
            qc_cols_pref = [
                c
                for c in [
                    "nuc_AreaShape_Area",
                    "act_AreaShape_Area",
                    "nuc_Intensity_MeanIntensity",
                ]
                if c in df.columns
            ]
            qc_groups = [c for c in ("cell_type", "treatment") if c in df.columns]
            df_flags = flag_outliers_per_group(
                df,
                group_cols=qc_groups,
                cols_mad_log=qc_cols_pref,
                z_cutoff=3.5,
                mitotic_rule=True,
            )
        else:
            # no QC: just add columns so downstream code is consistent
            df_flags = df.copy()
            df_flags["qc_keep"] = True
            df_flags["qc_reason"] = ""

        # summarize
        self.qc_summary = summarize_qc(
            df_flags,
            group_cols=[c for c in ("cell_type", "treatment") if c in df_flags.columns],
        )
        self.df_flags = df_flags.copy()

        # drop or keep
        df_clean = (
            df_flags[df_flags["qc_keep"]].reset_index(drop=True)
            if self.qc_drop
            else df_flags.reset_index(drop=True)
        )

        self.n_rows_before = int(df_flags.shape[0])
        self.n_rows_after = int(df_clean.shape[0])
        self.df_clean = df_clean  # <-- set FIRST
        self.df_trimmed = None  # reset
        self.cols = {}  # reset registry
        self.df = self.df_clean  # alias for backward compat

        # Normalization (add normalized copies; do not overwrite originals)
        self.df_norm = self.df_clean
        if self.normalize_mode in {"zscore", "robust"}:
            by = ["cell_type", "treatment"] if self.normalize_by_group else None
            self.df_norm = add_normalized_columns(
                self.df_clean,
                method=self.normalize_mode,
                by=by,
                suffix=self.norm_suffix,
                # optionally exclude even more columns by name:
                extra_exclude=[],
            )

    # remove unwanted feature columns
    def trim_features(self, feature_cols: List[str]) -> None:
        """Keep only meta columns + selected feature columns (or all numeric fallback)."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_and_clean() first.")
        meta = [
            c
            for c in ["ImageNumber", "ObjectNumber", "group", "FileName"]
            if c in self.df.columns
        ]
        feats = [c for c in feature_cols if c in self.df.columns]
        if not feats:
            feats = [
                c
                for c in self.df.select_dtypes(include=[np.number]).columns
                if c not in meta
            ]
        self.df_trimmed = self.df[meta + feats].copy()

    def run_stats(self) -> None:
        """
        Summary stats overall and by group (group = cell_type + '_' + treatment).
        Excludes metadata/ID columns from features.
        Produces:
        - overall
        - by_group_long
        - by_group_wide_mean
        - by_group_wide_mean_sem
        """
        # Choose the table in a way that narrows the type for Pylance
        if self.use_norm_for_stats and isinstance(self.df_norm, pd.DataFrame):
            df: pd.DataFrame = self.df_norm
        else:
            df = self.df_for_analysis()  # must return a DataFrame or raise

        # Ensure feature lists are populated (id/nuc/act)
        self._ensure_feature_lists()

        # --- choose numeric feature columns (drop metadata/IDs) ---
        id_set = set(self.cols.get("id", []))
        num_cols = [
            c for c in df.columns if c not in id_set and is_numeric_dtype(df[c])
        ]
        if not num_cols:
            self.stats_results = {
                "note": pd.DataFrame([{"msg": "No numeric features"}])
            }
            return
        num_df = df[num_cols].copy()

        # --- helper stats (typed & robust to non-numeric via coercion) ---
        def _to_num(x: pd.Series) -> pd.Series:
            return pd.to_numeric(x, errors="coerce")

        def sem(x: pd.Series) -> float:
            s = _to_num(x)
            n = int(s.count())
            return float(s.std(ddof=1) / np.sqrt(n)) if n > 0 else np.nan

        def ci_low(x: pd.Series) -> float:
            s = _to_num(x)
            n = int(s.count())
            if n <= 1:
                return np.nan
            m = float(s.mean())
            sd = float(s.std(ddof=1))
            if not np.isfinite(sd):
                return np.nan
            return float(m - 1.96 * sd / np.sqrt(n))

        def ci_high(x: pd.Series) -> float:
            s = _to_num(x)
            n = int(s.count())
            if n <= 1:
                return np.nan
            m = float(s.mean())
            sd = float(s.std(ddof=1))
            if not np.isfinite(sd):
                return np.nan
            return float(m + 1.96 * sd / np.sqrt(n))

        def iqr(x: pd.Series) -> float:
            s = _to_num(x).dropna()
            if s.empty:
                return np.nan
            arr = s.to_numpy(dtype=float)
            return float(np.percentile(arr, 75) - np.percentile(arr, 25))

        agg_funcs = ["count", "mean", "std", sem, ci_low, ci_high, "median", iqr]
        stat_names = [
            "count",
            "mean",
            "std",
            "sem",
            "ci_low",
            "ci_high",
            "median",
            "iqr",
        ]

        # --- overall (all rows) ---
        overall = num_df.agg(agg_funcs).T
        overall.columns = stat_names

        # Optional: rename to ci95_* for reporting
        # overall = overall.rename(columns={"ci_low": "ci95_low", "ci_high": "ci95_high"})

        results: Dict[str, pd.DataFrame] = {"overall": overall}

        # --- by group (e.g., HeLa_ctl, HeLa_ble, ...) ---
        if "group" in df.columns:
            g = df.groupby("group", observed=False)[num_cols].agg(
                agg_funcs
            )  # MultiIndex (feature, stat)

            # flatten MultiIndex to f"{feature}__{stat}"
            flat_cols = []
            for feat, fn in g.columns.to_list():
                name = fn if isinstance(fn, str) else fn.__name__
                flat_cols.append(f"{feat}__{name}")
            g_flat = g.copy()
            g_flat.columns = flat_cols
            g_flat = g_flat.reset_index()

            # tidy long -> (feature, group) x stats
            long = g_flat.melt(
                id_vars=["group"], var_name="feature_stat", value_name="value"
            )
            long[["feature", "stat"]] = long["feature_stat"].str.split(
                "__", n=1, expand=True
            )
            by_group_long = (
                long.pivot_table(
                    index=["feature", "group"],
                    columns="stat",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
                .reindex(columns=["feature", "group"] + stat_names)
            )
            by_group_long.columns.name = None

            # Optional: rename to ci95_* for reporting
            # by_group_long = by_group_long.rename(columns={"ci_low": "ci95_low", "ci_high": "ci95_high"})

            # wide mean
            by_group_wide_mean = by_group_long.pivot(
                index="feature", columns="group", values="mean"
            ).sort_index()

            # pretty "mean ± sem"
            mean_sem = (
                by_group_long["mean"].round(3).astype(str)
                + " ± "
                + by_group_long["sem"].round(3).astype(str)
            )
            tmp = by_group_long[["feature", "group"]].copy()
            tmp["mean_sem"] = mean_sem
            by_group_wide_mean_sem = tmp.pivot(
                index="feature", columns="group", values="mean_sem"
            ).sort_index()

            results["by_group_long"] = by_group_long
            results["by_group_wide_mean"] = by_group_wide_mean
            results["by_group_wide_mean_sem"] = by_group_wide_mean_sem

        self.stats_results = results

    def compare_vs_control(
        self,
        control_label: str = "ctl",
        group_col: str = "treatment",
        stratify_by: str | None = "cell_type",
    ) -> pd.DataFrame:
        """
        Compare each non-control group vs control for every numeric feature.
        Computes mean_ctl, mean_tx, diff, pct_change, Welch t-test p-value.
        Optionally stratifies by 'cell_type' (or set stratify_by=None).
        """
        df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
        if df is None:
            raise RuntimeError("No data loaded.")

        if group_col not in df.columns:
            raise ValueError(f"Column '{group_col}' not found in data.")

        # numeric features only (drop metadata)
        meta = {
            "ImageNumber",
            "ObjectNumber",
            "FileName",
            "group",
            "cell_type",
            "treatment",
        }
        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c not in meta
        ]
        if not numeric_cols:
            self.compare_results = pd.DataFrame()
            return self.compare_results

        def one_stratum(sub: pd.DataFrame, level_val: Any) -> pd.DataFrame:
            out_rows = []
            labels = sub[group_col].astype(str).str.lower()
            ctl_mask = labels == control_label.lower()
            if ctl_mask.sum() == 0:
                return pd.DataFrame()  # no control in this stratum

            for tx in sorted(labels.unique()):
                if tx == control_label.lower():
                    continue
                tx_mask = labels == tx
                for feat in numeric_cols:
                    a: pd.Series = _as_numeric_series(sub.loc[ctl_mask, feat]).dropna()
                    b: pd.Series = _as_numeric_series(sub.loc[tx_mask, feat]).dropna()

                    a_vals = a.to_numpy(dtype=float)
                    b_vals = b.to_numpy(dtype=float)
                    if a_vals.size < 2 or b_vals.size < 2:
                        t_stat, p_val = np.nan, np.nan
                    else:
                        t_stat, p_val = stats.ttest_ind(
                            b_vals, a_vals, equal_var=False, nan_policy="omit"
                        )

                    mean_ctl = float(np.nanmean(a_vals)) if a_vals.size else np.nan
                    mean_tx = float(np.nanmean(b_vals)) if b_vals.size else np.nan
                    diff = mean_tx - mean_ctl
                    pct = (
                        (diff / mean_ctl * 100.0)
                        if np.isfinite(mean_ctl) and mean_ctl != 0
                        else np.nan
                    )
                    row = {
                        "feature": feat,
                        group_col: tx,
                        "mean_ctl": mean_ctl,
                        "mean_tx": mean_tx,
                        "diff": diff,
                        "pct_change": pct,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "n_ctl": int(a_vals.size),
                        "n_tx": int(b_vals.size),
                    }
                    if stratify_by and stratify_by in sub.columns:
                        row[stratify_by] = level_val
                    out_rows.append(row)
            return pd.DataFrame(out_rows)

        if stratify_by and stratify_by in df.columns:
            parts = [
                one_stratum(g, lvl) for lvl, g in df.groupby(stratify_by, dropna=False)
            ]
            comp = (
                pd.concat([p for p in parts if not p.empty], ignore_index=True)
                if parts
                else pd.DataFrame()
            )
        else:
            comp = one_stratum(df, None)

        self.compare_results = comp
        return comp

    # ------------------- histograms - distributions -------------------

    def plot_group_distributions(
        self,
        columns: list[str],
        pretty_names: dict[str, str] | None = None,
        group_col: str = "treatment",
        filter_cell_type: str | None = None,  # e.g., "HeLa" or "NIH3T3"
        use_norm: bool = False,  # plot normalized copies if available
        save_dir: str | None = None,
        n_bootstrap: int = 2000,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        For each feature in `columns`, draw:
        - KDE histogram by treatment (ctl, ble, h11, y27)
        - Dumbbell plot with mean ± 95% CI
        Add cell line name in titles and filenames when `filter_cell_type` is set.
        Save figures and return a tidy CSV-like DataFrame with per-group stats.
        """

        # pick base table
        if use_norm and isinstance(self.df_norm, pd.DataFrame):
            df: pd.DataFrame = self.df_norm
        else:
            df = self.df_for_analysis()  # must return a DataFrame or raise

        # ensure group column exists
        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in data.")

        # optional restrict by cell type
        if filter_cell_type is not None and "cell_type" in df.columns:
            df = df[df["cell_type"].astype(str) == str(filter_cell_type)]

        # resolve requested columns (allow your old 'cell_' → new 'act_' mapping)
        cols = []
        for c in columns:
            cols.append("act_" + c[len("cell_") :] if c.startswith("cell_") else c)
        cols = [c for c in cols if c in df.columns]
        if not cols:
            raise ValueError("None of the requested columns exist in the DataFrame.")

        # palette / labels / order (lowercase keys, we’ll lower() group values)
        palette = {
            "ctl": "#AB6621",  # Tiger's eye
            "ble": "#375D6D",  # Payne's gray
            "h11": "#470A14",  # Chocolate cosmos
            "y27": "#5F5B44",  # Ebony
        }
        display = {
            "ctl": "Control",
            "ble": "Blebbistatin",
            "h11": "H-1152",
            "y27": "Y-27632",
        }
        order = ["ctl", "ble", "h11", "y27"]

        pretty_map = pretty_names or {}

        # output dir
        if save_dir is None:
            base_out = getattr(self, "output_dir", "outputs")
            save_dir = os.path.join(base_out, "figures")
        os.makedirs(save_dir, exist_ok=True)

        # style
        plt.rcParams.update(
            {
                "font.size": 14,
                "font.family": "serif",
                "axes.titlesize": 18,
                "axes.labelsize": 16,
                "legend.fontsize": 10,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        )
        sns.set_style("whitegrid")

        rng = np.random.default_rng(random_state)

        def bootstrap_ci(
            values: np.ndarray, func=np.nanmean, n=n_bootstrap
        ) -> tuple[float, float]:
            values = values[np.isfinite(values)]
            if values.size == 0:
                return (np.nan, np.nan)
            idx = rng.integers(0, values.size, size=(n, values.size))
            stats_boot = func(values[idx], axis=1)
            return float(np.percentile(stats_boot, 2.5)), float(
                np.percentile(stats_boot, 97.5)
            )

        def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
            a = a[np.isfinite(a)]
            b = b[np.isfinite(b)]
            if a.size < 2 or b.size < 2:
                return np.nan
            nx, ny = a.size, b.size
            dof = nx + ny - 2
            if dof <= 0:
                return np.nan
            s = np.sqrt(
                ((nx - 1) * np.var(a, ddof=1) + (ny - 1) * np.var(b, ddof=1)) / dof
            )
            return float((np.mean(a) - np.mean(b)) / s) if s > 0 else np.nan

        # helpers for titles/filenames when filtering a cell line
        ct_prefix = f"{filter_cell_type} — " if filter_cell_type else ""
        ct_suffix = f"_{filter_cell_type}" if filter_cell_type else ""

        all_rows: list[dict] = []

        # lower-cased group values used for selection
        gvals = df[group_col].astype(str).str.lower()

        for col in cols:
            label = pretty_map.get(col, col)

            # collect per-treatment arrays
            series_by_group: dict[str, np.ndarray] = {}
            for g in order:
                vals = pd.to_numeric(df.loc[gvals == g, col], errors="coerce").to_numpy(
                    dtype=float
                )
                series_by_group[g] = vals

            # KDE histogram
            plt.figure(figsize=(10, 6))
            for g in order:
                vals = series_by_group[g]
                if vals.size == 0:
                    continue
                sns.kdeplot(
                    vals[np.isfinite(vals)],
                    fill=True,
                    color=palette[g],
                    label=display[g],
                    linewidth=1.5,
                    alpha=0.6,
                )
                m = float(np.nanmean(vals)) if vals.size else np.nan
                if np.isfinite(m):
                    plt.axvline(m, color=palette[g], linestyle="--", linewidth=1)
            plt.title(f"{ct_prefix}Distribution of {label}")
            plt.xlabel(label)
            plt.ylabel("Density")
            plt.legend(title="Group")
            plt.grid(False)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", col)
            plt.savefig(
                os.path.join(save_dir, f"Hist_{safe}_byGroup{ct_suffix}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # Stats + dumbbell
            means, cvs, ci_lo, ci_hi, pvals, effects, ns = {}, {}, {}, {}, {}, {}, {}
            ctl_vals = series_by_group["ctl"]

            for g in order:
                vals = series_by_group[g]
                mu = float(np.nanmean(vals)) if vals.size else np.nan
                sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else np.nan
                cv = (
                    (sd / mu)
                    if (np.isfinite(sd) and np.isfinite(mu) and mu != 0)
                    else np.nan
                )
                lo, hi = bootstrap_ci(vals)

                means[g], cvs[g], ci_lo[g], ci_hi[g], ns[g] = (
                    mu,
                    cv,
                    lo,
                    hi,
                    int(vals.size),
                )

                if g == "ctl":
                    pvals[g], effects[g] = np.nan, np.nan
                else:
                    if ctl_vals.size >= 2 and vals.size >= 2:
                        t, p = stats.ttest_ind(
                            vals[np.isfinite(vals)],
                            ctl_vals[np.isfinite(ctl_vals)],
                            equal_var=False,
                            nan_policy="omit",
                        )
                        pvals[g] = float(p)
                        effects[g] = cohen_d(ctl_vals, vals)
                    else:
                        pvals[g], effects[g] = np.nan, np.nan

                # add summary row
                all_rows.append(
                    {
                        "feature": col,
                        "label": label,
                        "group": g,
                        "group_label": display[g],
                        "cell_type": (filter_cell_type or ""),
                        "n": ns[g],
                        "mean": means[g],
                        "std": sd,
                        "cv": cvs[g],
                        "ci95_low": ci_lo[g],
                        "ci95_high": ci_hi[g],
                        "p_vs_ctl": pvals[g],
                        "cohen_d_vs_ctl": effects[g],
                        "used_norm": bool(use_norm),
                    }
                )

            # dumbbell plot
            plt.figure(figsize=(3.8, 5.0))
            for g in order:
                if not np.isfinite(means[g]):
                    continue
                plt.errorbar(
                    y=means[g],
                    x=display[g],
                    yerr=[[means[g] - ci_lo[g]], [ci_hi[g] - means[g]]],
                    fmt="o",
                    color=palette[g],
                    capsize=5,
                    markersize=9,
                    linewidth=2,
                    capthick=2,
                )
            xs = [display[g] for g in order if np.isfinite(means[g])]
            ys = [means[g] for g in order if np.isfinite(means[g])]
            if len(xs) >= 2:
                plt.plot(xs, ys, color="gray", linestyle="--", linewidth=1)

            plt.ylabel(label)
            plt.xlabel("")
            plt.title(f"{ct_prefix}Mean {label} by Group (95% CI)")
            plt.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f"Dumbbell_{safe}_byGroup{ct_suffix}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        summary = pd.DataFrame(all_rows)
        out_csv = os.path.join(save_dir, f"plots_summary{ct_suffix or ''}.csv")
        summary.to_csv(out_csv, index=False)
        return summary

    def plot_by_celltype(
        self,
        columns: list[str],
        pretty_names: dict[str, str] | None = None,
        group_col: str = "treatment",
        use_norm: bool = False,
        save_dir: str | None = None,
        n_bootstrap: int = 2000,
        random_state: int = 0,
    ) -> pd.DataFrame:
        """
        Convenience wrapper: run `plot_group_distributions` once per cell_type.
        Each figure filename and title includes the cell line.
        Returns one combined summary DataFrame.
        """
        import os

        df = (
            self.df_norm
            if (use_norm and getattr(self, "df_norm", None) is not None)
            else self.df_for_analysis()
        )

        if "cell_type" not in df.columns:
            # no cell_type present: single run
            return self.plot_group_distributions(
                columns=columns,
                pretty_names=pretty_names,
                group_col=group_col,
                filter_cell_type=None,
                use_norm=use_norm,
                save_dir=save_dir,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

        # unique, stable order of cell types
        if pd.api.types.is_categorical_dtype(df["cell_type"]):
            levels = [str(x) for x in df["cell_type"].cat.categories]
        else:
            levels = sorted(df["cell_type"].dropna().astype(str).unique())

        base_dir = save_dir or os.path.join(
            getattr(self, "output_dir", "outputs"), "figures"
        )
        os.makedirs(base_dir, exist_ok=True)

        parts = []
        for ct in levels:
            # put all figures in the same folder, but suffix filenames with cell line
            s = self.plot_group_distributions(
                columns=columns,
                pretty_names=pretty_names,
                group_col=group_col,
                filter_cell_type=ct,
                use_norm=use_norm,
                save_dir=base_dir,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if "cell_type" not in s.columns:
                s["cell_type"] = ct
            parts.append(s)

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # ------------------- regression and random forest methods -------------------

    def run_regression(self) -> None:
        """
        Toy least-squares regression:
        - target = first numeric column
        - predictors = remaining numeric columns
        Returns coefficients and R^2. This is just a smoke-test regressor.
        """
        df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
        if df is None:
            raise RuntimeError("No data available. Did you run load_and_clean()?")

        # choose numeric columns (exclude metadata)
        meta = {
            "ImageNumber",
            "ObjectNumber",
            "FileName",
            "group",
            "cell_type",
            "treatment",
        }
        numeric_cols = [
            c for c in df.columns if is_numeric_dtype(df[c]) and c not in meta
        ]

        if len(numeric_cols) < 2:
            self.regression_results = {
                "meta": pd.DataFrame(
                    [{"note": "Insufficient numeric features for regression."}]
                )
            }
            return

        y_col = numeric_cols[0]
        X_cols = numeric_cols[1:]

        # coerce to numeric and to NumPy arrays
        y: NDArray[np.float64] = pd.to_numeric(df[y_col], errors="coerce").to_numpy(
            dtype=float
        )
        X: NDArray[np.float64] = (
            df[X_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        )

        # drop rows with any NaN/inf
        valid_mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        y = y[valid_mask]
        X = X[valid_mask]

        if y.size < 2 or X.shape[1] == 0 or X.shape[0] != y.size:
            self.regression_results = {
                "meta": pd.DataFrame(
                    [{"note": "Not enough valid rows after cleaning."}]
                )
            }
            return

        # add intercept column
        ones = np.ones((X.shape[0], 1), dtype=float)
        X_design: NDArray[np.float64] = np.concatenate([ones, X], axis=1)

        # solve least squares y ~ [1, X] beta
        beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)

        # fit stats
        y_hat = X_design @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - y_hat) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        coef_df = pd.DataFrame({"term": ["intercept"] + X_cols, "coef": beta})
        metrics_df = pd.DataFrame([{"target": y_col, "r2": r2, "n": int(y.size)}])

        self.regression_results = {"coefficients": coef_df, "metrics": metrics_df}

    def run_random_forest(self) -> None:
        """
        Importance proxy without sklearn:
        - target = first numeric column
        - features = remaining numeric columns
        - importance = |Pearson correlation(target, feature)|
        """
        df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
        if df is None:
            raise RuntimeError("No data available. Did you run load_and_clean()?")

        # choose numeric columns; drop metadata
        meta = {
            "ImageNumber",
            "ObjectNumber",
            "FileName",
            "group",
            "cell_type",
            "treatment",
        }
        numeric_cols = [
            c for c in df.columns if is_numeric_dtype(df[c]) and c not in meta
        ]

        if len(numeric_cols) < 2:
            self.random_forest_results = pd.DataFrame(
                [{"note": "Insufficient numeric features for importance proxy."}]
            )
            return

        y_col = numeric_cols[0]
        X_cols = numeric_cols[1:]

        # numeric arrays
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
        valid_y = np.isfinite(y)

        rows = []
        for c in X_cols:
            x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            mask = valid_y & np.isfinite(x)
            if mask.sum() < 2:
                corr = np.nan
            else:
                corr = float(np.corrcoef(y[mask], x[mask])[0, 1])
            rows.append(
                {
                    "target": y_col,
                    "feature": c,
                    "corr": corr,
                    "abs_corr": abs(corr) if np.isfinite(corr) else np.nan,
                    "n_used": int(mask.sum()),
                }
            )

        imp = (
            pd.DataFrame(rows)
            .sort_values("abs_corr", ascending=False, na_position="last")
            .reset_index(drop=True)
        )
        self.random_forest_results = imp

    def get_data_summary(self) -> str:
        """Provide a short text summary for debugging."""
        parts: List[str] = []
        for name, df in [("df_clean", self.df_clean), ("df_trimmed", self.df_trimmed)]:
            if df is None:
                parts.append(f"{name}: None")
            else:
                cols_preview = list(df.columns)[:8]
                suffix = "..." if df.shape[1] > 8 else ""
                parts.append(f"{name}: shape={df.shape}, cols={cols_preview}{suffix}")
        return " | ".join(parts)


# -------------------------- utilities --------------------------
