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
from scipy import stats
from typing import List, Optional, Dict, Any, Sequence, Union
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from pandas import CategoricalDtype

import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import pandas as pd


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

        self.qc_mode = qc_mode
        self.qc_drop = qc_drop
        self.add_aspect_ratio_flag = add_aspect_ratio

        self.problem_files_path = problem_files_path
        self.file_column = file_column
        self.pixel_size_um = pixel_size_um
        self.convert_coordinates = convert_coordinates

        self.df: Optional[pd.DataFrame] = None

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
        self.output_dir: Optional[str] = None
        self.stats_results: Optional[Dict[str, pd.DataFrame]] = None
        self.regression_results: Optional[
            Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        ] = None
        self.random_forest_results: Optional[pd.DataFrame] = None
        self.compare_results: Optional[pd.DataFrame] = None

    # ------------------- helpers for methods -------------------

    def df_for_analysis(self) -> pd.DataFrame:
        """Return a DataFrame for analysis, or raise if none available."""
        df = (
            self.df_trimmed
            if isinstance(self.df_trimmed, pd.DataFrame)
            else self.df_clean
        )
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError("No data available; run load_and_clean().")
        return df

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

    def _df_analysis(
        self,
        use_norm: bool = False,
        filter_cell_types: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Return the table to analyze, optionally normalized and/or filtered by cell_type."""
        df = (
            self.df_norm
            if (use_norm and isinstance(self.df_norm, pd.DataFrame))
            else self.df_for_analysis()
        )
        if filter_cell_types and "cell_type" in df.columns:
            wanted = [str(x) for x in filter_cell_types]
            df = df[df["cell_type"].astype(str).isin(wanted)]
        return df

    def _coerce_numeric_df(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df[cols].apply(pd.to_numeric, errors="coerce").astype(float)

    def _impute_with_median(self, X: pd.DataFrame) -> pd.DataFrame:
        med = X.median(axis=0, numeric_only=True)
        return X.fillna(med)

    def _std_scale(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0, ddof=1)
        std[~np.isfinite(std)] = 1.0
        std[std == 0.0] = 1.0
        return (X - mean) / std, mean, std

    # ------------------- data cleaning methods in this class-------------------

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
        """Keep only metadata + selected feature columns (or all numeric fallback)."""
        if self.df is None:
            raise RuntimeError("Data not loaded. Call load_and_clean() first.")

        # meta columns we ALWAYS keep in trimmed tables
        always_keep = [
            c
            for c in [
                "ImageNumber",
                "ObjectNumber",
                "FileName",
                "group",  # composite label (cell_type + '_' + treatment)
                "cell_type",  # needed by plot_by_celltype and comparisons
                "treatment",  # needed by plotting & comparisons
            ]
            if c in self.df.columns
        ]

        # the features the user asked for (if present)
        feats = [c for c in feature_cols if c in self.df.columns]

        # if none provided, keep all numeric columns EXCEPT meta
        if not feats:
            numeric = list(self.df.select_dtypes(include=[np.number]).columns)
            feats = [c for c in numeric if c not in always_keep]

        # build trimmed view
        self.df_trimmed = self.df[always_keep + feats].copy()

    # ------------------- statistical analysis methods in this class-------------------

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
        # pick DF
        df: pd.DataFrame = (
            self.df_norm
            if (self.use_norm_for_stats and isinstance(self.df_norm, pd.DataFrame))
            else self.df_for_analysis()
        )
        self._ensure_feature_lists()

        # numeric feature columns (drop IDs/metadata)
        self._ensure_feature_lists()
        id_set = set(self.cols.get("id", []))

        num_cols = [
            c for c in df.columns
            if c not in id_set
            and is_numeric_dtype(df[c])
            and not is_bool_dtype(df[c])  # exclude boolean flags like qc_keep
        ]

        if not num_cols:
            self.stats_results = {"note": pd.DataFrame([{"msg": "No numeric features"}])}
            return

        num_df = df[num_cols].copy()
        # optional: drop columns that are entirely NaN to avoid “mean of empty slice” spam
        num_df = num_df.loc[:, num_df.notna().any(axis=0)]

        # ---------- OVERALL ----------
        # basic reducers first (no custom functions here)
        basic = num_df.agg(["count", "mean", "std", "median"]).T

        # sem (as Series aligned to basic.index)
        def _sem_col(s: pd.Series) -> float:
            s = pd.to_numeric(s, errors="coerce")
            n = int(s.count())
            return float(s.std(ddof=1) / np.sqrt(n)) if n > 0 else np.nan

        sem_s = num_df.apply(_sem_col, axis=0).reindex(basic.index)

        # IQR from separate quantiles (avoids list[float] stub issue)
        q25 = num_df.quantile(0.25, numeric_only=True).reindex(basic.index)
        q75 = num_df.quantile(0.75, numeric_only=True).reindex(basic.index)
        iqr_s = q75 - q25

        ci95_low = basic["mean"] - 1.96 * sem_s
        ci95_high = basic["mean"] + 1.96 * sem_s

        overall = pd.DataFrame(
            {
                "count": pd.Series(basic["count"].astype(float), index=basic.index),
                "mean": pd.Series(basic["mean"], index=basic.index),
                "std": pd.Series(basic["std"], index=basic.index),
                "sem": pd.Series(sem_s, index=basic.index),
                "ci95_low": pd.Series(ci95_low, index=basic.index),
                "ci95_high": pd.Series(ci95_high, index=basic.index),
                "median": pd.Series(basic["median"], index=basic.index),
                "iqr": pd.Series(iqr_s, index=basic.index),
            }
        )

        results: Dict[str, pd.DataFrame] = {"overall": overall}

        # ---------- BY GROUP ----------
        if "group" in df.columns:
            g = df.groupby("group", observed=False)

            mean_g = g[num_cols].mean()
            count_g = g[num_cols].count()
            std_g = g[num_cols].std(ddof=1)
            median_g = g[num_cols].median()

            # sem per group
            sem_g = std_g.div(count_g.replace(0, np.nan).pow(0.5))

            # IQR via separate quantiles (no list)
            q25_g = g[num_cols].quantile(0.25)
            q75_g = g[num_cols].quantile(0.75)
            iqr_g = q75_g - q25_g

            ci95_low_g = mean_g - 1.96 * sem_g
            ci95_high_g = mean_g + 1.96 * sem_g

            # build tidy long by stacking each table and merging
            from functools import reduce

            def _stack(name: str, d: pd.DataFrame) -> pd.DataFrame:
                s = d.stack()
                s.name = name  # <- avoid Series.rename typing issue
                out = s.reset_index().rename(columns={"level_1": "feature"})
                return out

            parts = [
                _stack("count", count_g),
                _stack("mean", mean_g),
                _stack("std", std_g),
                _stack("sem", sem_g),
                _stack("ci95_low", ci95_low_g),
                _stack("ci95_high", ci95_high_g),
                _stack("median", median_g),
                _stack("iqr", iqr_g),
            ]
            by_group_long = reduce(
                lambda L, R: pd.merge(L, R, on=["group", "feature"], how="outer"), parts
            )

            # wide mean and pretty mean±sem
            by_group_wide_mean = mean_g.T.sort_index()
            by_group_wide_mean.index.name = "feature"

            tmp = by_group_long[["feature", "group"]].copy()
            tmp["mean_sem"] = (
                by_group_long["mean"].round(3).astype(str)
                + " ± "
                + by_group_long["sem"].round(3).astype(str)
            )
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
        For each feature in `columns`, draw KDEs by treatment and a 'dumbbell' mean±CI plot.
        Returns a tidy DataFrame of per-group stats and saves figures.
        """
        # choose DataFrame
        if use_norm and isinstance(self.df_norm, pd.DataFrame):
            df: pd.DataFrame = self.df_norm
        else:
            df = self.df_for_analysis()

        if group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found in data.")

        # optional cell-type restriction
        if filter_cell_type is not None and "cell_type" in df.columns:
            df = df[df["cell_type"].astype(str) == str(filter_cell_type)]

        # map old 'cell_' to 'act_' prefix and keep only existing columns
        col_list: list[str] = []
        for c in columns:
            col_list.append("act_" + c[len("cell_") :] if c.startswith("cell_") else c)
        col_list = [c for c in col_list if c in df.columns]
        if not col_list:
            raise ValueError("None of the requested columns exist in the DataFrame.")

        # palette / labels / consistent order
        palette = {
            "ctl": "#AB6621",
            "ble": "#375D6D",
            "h11": "#470A14",
            "y27": "#5F5B44",
        }
        display = {
            "ctl": "Control",
            "ble": "Blebbistatin",
            "h11": "H-1152",
            "y27": "Y-27632",
        }
        order = ["ctl", "ble", "h11", "y27"]
        pretty_map = pretty_names or {}

        # output directory
        if save_dir is None:
            base_out = getattr(self, "output_dir", "outputs")
            save_dir_path: str = os.path.join(base_out, "figures")
        else:
            save_dir_path = str(save_dir)
        os.makedirs(save_dir_path, exist_ok=True)

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
            vv = values[np.isfinite(values)]
            if vv.size == 0:
                return (np.nan, np.nan)
            idx = rng.integers(0, vv.size, size=(n, vv.size))
            stats_boot = func(vv[idx], axis=1)
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

        # suffix/prefix used in titles/filenames
        ct_prefix = f"{filter_cell_type} — " if filter_cell_type else ""
        ct_suffix = f"_{filter_cell_type}" if filter_cell_type else ""

        # summary rows accumulator
        all_rows: list[dict[str, Any]] = []

        # cached lower-cased group labels for selection
        gvals = df[group_col].astype(str).str.lower()

        for col in col_list:
            label = pretty_map.get(col, col)

            # per-treatment arrays
            series_by_group: dict[str, np.ndarray] = {}
            for g in order:
                s: pd.Series = df.loc[gvals == g, col]  # ensure Series
                vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
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
                os.path.join(save_dir_path, f"Hist_{safe}_byGroup{ct_suffix}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

            # stats & summaries per group
            means: dict[str, float] = {}
            cvs: dict[str, float] = {}
            ci_lo: dict[str, float] = {}
            ci_hi: dict[str, float] = {}
            pvals: dict[str, float] = {}
            effects: dict[str, float] = {}
            ns: dict[str, int] = {}

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
                        res = stats.ttest_ind(
                            vals[np.isfinite(vals)],
                            ctl_vals[np.isfinite(ctl_vals)],
                            equal_var=False,
                            nan_policy="omit",
                        )
                        p_any = getattr(res, "pvalue", None)
                        if p_any is None:  # tuple-like result
                            p_any = res[1]  # type: ignore[index]
                        pvals[g] = float(np.asarray(p_any).item())

                        effects[g] = cohen_d(ctl_vals, vals)
                    else:
                        pvals[g], effects[g] = np.nan, np.nan

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

            # Dumbbell plot with numeric x positions (keeps type checker happy)
            fig, ax = plt.subplots(figsize=(3.8, 5.0))
            xs = []
            ys = []
            for i, g in enumerate(order):
                if not np.isfinite(means[g]):
                    continue
                xs.append(i)
                ys.append(means[g])
                yerr = np.array(
                    [[means[g] - ci_lo[g]], [ci_hi[g] - means[g]]], dtype=float
                )
                ax.errorbar(
                    i,
                    means[g],
                    yerr=yerr,
                    fmt="o",
                    color=palette[g],
                    capsize=5,
                    markersize=9,
                    linewidth=2,
                    capthick=2,
                )
            if len(xs) >= 2:
                ax.plot(xs, ys, color="gray", linestyle="--", linewidth=1)

            ax.set_ylabel(label)
            ax.set_xlabel("")
            ax.set_title(f"{ct_prefix}Mean {label} by Group (95% CI)")
            ax.grid(True, axis="y", alpha=0.3)
            xticks = np.arange(float(len(order)), dtype=float)  # 0.0, 1.0, 2.0, 3.0
            ax.set_xticks(xticks.tolist())
            ax.set_xticklabels([display[g] for g in order], rotation=30)
            fig.tight_layout()
            fig.savefig(
                os.path.join(save_dir_path, f"Dumbbell_{safe}_byGroup{ct_suffix}.png"),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

        summary = pd.DataFrame(all_rows)
        out_csv = os.path.join(save_dir_path, f"plots_summary{ct_suffix or ''}.csv")
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
        Run `plot_group_distributions` once per cell_type.
        Filenames/titles include the cell line. Returns a combined summary DataFrame.
        """
        if use_norm and isinstance(self.df_norm, pd.DataFrame):
            df: pd.DataFrame = self.df_norm
        else:
            df = self.df_for_analysis()

        if "cell_type" not in df.columns:
            res = self.plot_group_distributions(
                columns=columns,
                pretty_names=pretty_names,
                group_col=group_col,
                filter_cell_type=None,
                use_norm=use_norm,
                save_dir=save_dir,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            return res

        if isinstance(df["cell_type"].dtype, CategoricalDtype):
            levels = [str(x) for x in df["cell_type"].cat.categories]
        else:
            levels = sorted(df["cell_type"].dropna().astype(str).unique())

        base_dir = save_dir or os.path.join(
            getattr(self, "output_dir", "outputs"), "figures"
        )
        os.makedirs(base_dir, exist_ok=True)

        parts: list[pd.DataFrame] = []
        for ct in levels:
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

    # ------------------- PCA and dimensionality reduction methods -------------------

    def run_pca(
        self,
        predictors: str = "act",                # "act" → cell features, "nuc" → nuclear features
        use_norm: bool = True,                  # use normalized copy if available
        filter_cell_types: list[str] | None = None,  # run on all rows or per provided cell types
        n_components: int | None = None,        # None = all components
        var_tol: float = 1e-12,                 # drop near-constant columns
    ) -> None:
        """
        Robust PCA on chosen predictor feature block. Cleans data, prevents SVD failures,
        and falls back to covariance eigen-decomposition if needed.
        Saves results in self.pca_results (overall) and self.pca_by_celltype (optional).
        """
        # pick base DataFrame
        base = self.df_norm if (use_norm and isinstance(self.df_norm, pd.DataFrame)) else self.df_for_analysis()

        # ensure feature lists exist
        self._ensure_feature_lists()
        if predictors.lower() == "act":
            feat_cols = list(self.cols.get("act", []))
        elif predictors.lower() == "nuc":
            feat_cols = list(self.cols.get("nuc", []))
        else:
            raise ValueError("predictors must be 'act' or 'nuc'")

        # keep only numeric, non-boolean columns that exist
        feat_cols = [c for c in feat_cols if c in base.columns and is_numeric_dtype(base[c]) and not is_bool_dtype(base[c])]
        if not feat_cols:
            print("[PCA] No numeric predictor columns found after filtering.")
            self.pca_results = {}
            return

        def _clean_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, dict]:
            """Return (Xs, cols_kept, mean, std, info) with robust cleaning."""
            Xdf = df[feat_cols].copy()

            # drop columns that are entirely NaN
            all_nan = Xdf.isna().all(axis=0)
            if all_nan.any():
                Xdf = Xdf.loc[:, ~all_nan]

            # drop boolean cols defensively (shouldn't exist now)
            Xdf = Xdf.loc[:, [c for c in Xdf.columns if not is_bool_dtype(Xdf[c])]]

            # if nothing left
            if Xdf.shape[1] == 0:
                return np.empty((0, 0)), [], np.array([]), np.array([]), {"dropped_all_nan": int(all_nan.sum()), "dropped_low_var": 0}

            # impute NaNs by column median (numeric)
            med = Xdf.median(numeric_only=True)
            Xdf = Xdf.fillna(med)

            # drop near-constant columns
            std0 = Xdf.std(ddof=0)
            low_var_mask = (std0.fillna(0.0).to_numpy() <= var_tol)
            if low_var_mask.any():
                Xdf = Xdf.loc[:, ~low_var_mask]
                std0 = Xdf.std(ddof=0)

            cols_kept = list(Xdf.columns)

            # center / scale with epsilon
            mean = Xdf.mean().to_numpy(dtype=float)
            std = std0.to_numpy(dtype=float)
            eps = 1e-12
            std_safe = np.where(std > 0, std, 1.0)

            X = Xdf.to_numpy(dtype=float)
            if X.size == 0:
                return np.empty((0, 0)), [], np.array([]), np.array([]), {
                    "dropped_all_nan": int(all_nan.sum()),
                    "dropped_low_var": int(low_var_mask.sum()) if isinstance(low_var_mask, np.ndarray) else 0
                }

            Xs = (X - mean) / (std_safe + eps)

            info = {
                "dropped_all_nan": int(all_nan.sum()),
                "dropped_low_var": int((~np.isin(feat_cols, cols_kept)).sum()),
                "n_rows_used": int(Xs.shape[0]),
                "n_cols_used": int(Xs.shape[1]),
            }
            return Xs, cols_kept, mean, std_safe, info

        def _pca_from_matrix(Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Compute PCA; try SVD, fall back to covariance eigendecomp on failure."""
            if Xs.ndim != 2 or Xs.shape[0] < 2 or Xs.shape[1] < 2:
                raise ValueError("Need at least 2 rows and 2 columns for PCA.")
            try:
                U, S, Vt = np.linalg.svd(Xs, full_matrices=False)  # shapes: (n,k), (k,), (k,k)
                return U, S, Vt
            except np.linalg.LinAlgError:
                # fallback: eigen-decomposition of covariance
                C = (Xs.T @ Xs) / max(1, (Xs.shape[0] - 1))  # (k,k) symmetric
                w, V = np.linalg.eigh(C)                     # ascending
                idx = np.argsort(w)[::-1]                    # descending
                w = w[idx]
                V = V[:, idx]
                S = np.sqrt(np.clip(w * max(1, (Xs.shape[0] - 1)), a_min=0.0, a_max=None))
                U = Xs @ V @ np.diag(1.0 / (S + 1e-12))      # back out scores-like U (n,k)
                Vt = V.T
                return U, S, Vt

        def _package_results(Xs, U, S, Vt, cols, mean, std) -> dict:
            n = Xs.shape[0]
            var = (S ** 2) / max(1, (n - 1))
            total = float(var.sum()) if var.size else 0.0
            ratio = var / total if total > 0 else np.zeros_like(var)
            if n_components is not None:
                k = min(n_components, S.size)
                U = U[:, :k]; S = S[:k]; Vt = Vt[:k, :]; var = var[:k]; ratio = ratio[:k]
            scores = U * S  # (n, k)
            return {
                "X_cols": cols,
                "mean": mean,
                "std": std,
                "Vt": Vt,
                "scores": scores,
                "explained_variance": var,
                "explained_variance_ratio": ratio,
            }

        # --- run overall PCA (no per-cell-type filtering) ---
        Xs, cols_kept, mean, std, info = _clean_matrix(base)
        if Xs.shape[0] >= 2 and Xs.shape[1] >= 2:
            U, S, Vt = _pca_from_matrix(Xs)
            self.pca_results = _package_results(Xs, U, S, Vt, cols_kept, mean, std)
        else:
            self.pca_results = {}
            print(f"[PCA] Skipped overall PCA: insufficient data (rows={Xs.shape[0]}, cols={Xs.shape[1]}).")

        # --- optionally run PCA per cell type ---
        self.pca_by_celltype: dict[str, dict] = {}
        if filter_cell_types and "cell_type" in base.columns:
            for ct in filter_cell_types:
                sub = base[base["cell_type"].astype(str) == str(ct)]
                Xs_ct, cols_ct, mean_ct, std_ct, info_ct = _clean_matrix(sub)
                if Xs_ct.shape[0] >= 2 and Xs_ct.shape[1] >= 2:
                    try:
                        U_ct, S_ct, Vt_ct = _pca_from_matrix(Xs_ct)
                        self.pca_by_celltype[ct] = _package_results(Xs_ct, U_ct, S_ct, Vt_ct, cols_ct, mean_ct, std_ct)
                    except Exception as e:
                        print(f"[PCA] Skipped {ct}: {e}")
                else:
                    print(f"[PCA] Skipped {ct}: insufficient data (rows={Xs_ct.shape[0]}, cols={Xs_ct.shape[1]}).")


    # ------------------- regression and random forest methods -------------------

    def run_regression(
        self,
        targets: str = "nuc",  # predict nuclear features
        predictors: str = "act",  # from cell features
        n_components: int | None = None,
        var_threshold: float | None = 0.95,  # keep PCs up to this variance
        use_norm: bool = False,
        filter_cell_types: Sequence[str] | None = None,
    ) -> None:
        """Regress each target feature on PCA scores of predictor features."""
        # make sure PCA is ready and matches config
        need_fit = (
            getattr(self, "pca_results", None) is None
            or self.pca_results.get("predictor_set") != predictors
            or self.pca_results.get("used_norm") != bool(use_norm)
            or (
                filter_cell_types is not None
                and self.pca_results.get("filtered_cell_types")
                != list(filter_cell_types)
            )
        )
        if need_fit:
            self.run_pca(
                predictors=predictors,
                use_norm=use_norm,
                filter_cell_types=filter_cell_types,
            )

        df = self._df_analysis(use_norm=use_norm, filter_cell_types=filter_cell_types)
        self._ensure_feature_lists()
        Y_cols_all = self.cols.get(targets, [])
        Y_cols = [c for c in Y_cols_all if c in df.columns and is_numeric_dtype(df[c])]
        if not Y_cols:
            raise RuntimeError(f"No numeric target columns found for '{targets}'.")

        X_cols = self.pca_results["X_cols"]
        X_df = self._coerce_numeric_df(df, X_cols)
        X_df = self._impute_with_median(X_df)
        X = X_df.to_numpy(dtype=float)

        Vt: np.ndarray = self.pca_results["Vt"]
        var_ratio: np.ndarray = self.pca_results["explained_variance_ratio"]
        scores_all: np.ndarray = self.pca_results["scores"]
        std_pred: np.ndarray = self.pca_results["std"]

        # choose number of PCs
        if n_components is None:
            var_threshold = 0.95 if var_threshold is None else float(var_threshold)
            k = int(
                np.searchsorted(np.cumsum(var_ratio), var_threshold, side="left") + 1
            )
        else:
            k = int(max(1, min(n_components, Vt.shape[0])))

        V_k = Vt[:k, :].T  # (n_features x k)
        Z = scores_all[:, :k]  # (m x k)
        var_expl = float(var_ratio[:k].sum())

        coef_rows: list[dict[str, float | str | int]] = []
        met_rows: list[dict[str, float | str | int]] = []

        for y_name in Y_cols:
            y = pd.to_numeric(df[y_name], errors="coerce").astype(float).to_numpy()
            mask = np.isfinite(y)
            if mask.sum() < (k + 2):
                met_rows.append(
                    {
                        "target": y_name,
                        "r2": np.nan,
                        "n": int(mask.sum()),
                        "n_components": k,
                        "var_explained": var_expl,
                    }
                )
                continue

            y_m = y[mask]
            Z_m = Z[mask, :]
            X_m = X[mask, :]

            # y = a + Z_m @ gamma  (least squares)
            A = np.column_stack([np.ones(Z_m.shape[0], dtype=float), Z_m])
            beta_pc, _, _, _ = np.linalg.lstsq(A, y_m, rcond=None)
            gamma = beta_pc[1:]  # (k,)

            # back-transform PC coefs to original predictor features
            beta_std = V_k @ gamma  # on standardized X
            beta_orig = beta_std / std_pred  # original X units

            # intercept in original space
            mu_x = X_m.mean(axis=0)
            intercept = float(y_m.mean() - float(np.dot(beta_orig, mu_x)))

            # R^2
            y_hat = intercept + X_m @ beta_orig
            ss_tot = float(np.sum((y_m - y_m.mean()) ** 2))
            ss_res = float(np.sum((y_m - y_hat) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            # store
            coef_rows.append({"target": y_name, "term": "intercept", "coef": intercept})
            for feat, b in zip(X_cols, beta_orig, strict=False):
                coef_rows.append({"target": y_name, "term": feat, "coef": float(b)})

            met_rows.append(
                {
                    "target": y_name,
                    "r2": r2,
                    "n": int(y_m.size),
                    "n_components": int(k),
                    "var_explained": var_expl,
                }
            )

        coef_df = pd.DataFrame(coef_rows)
        met_df = pd.DataFrame(met_rows)

        if self.regression_results is None:
            self.regression_results = {}
        self.regression_results["pca"] = {"coefficients": coef_df, "metrics": met_df}

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
