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
import os
from typing import List, Optional, Dict
import numpy as np
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
    CONVERT_COORDINATES_DEFAULT,
    flag_outliers_per_group,
)

# -------------------define the main analysis class-------------------


class Nuclear_Shape_Analysis:
    def __init__(
        self,
        nuc_path: str,
        act_path: str,
        keep_columns: Optional[List[str]] = None,
        problem_files_path: Optional[str] = None,
        file_column: str = "FileName_Act",
        pixel_size_um: Optional[float] = None,
        convert_coordinates: bool = CONVERT_COORDINATES_DEFAULT,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        default_cell_type: str | None = None,
        default_treatment: str | None = None,
        label_map_path: str | None = None,
    ):
        self.nuc_path = nuc_path
        self.act_path = act_path
        self.keep_columns = keep_columns or [
            "ImageNumber",
            "ObjectNumber",
            "FileName_Nuc",
            "FileName_Act",
        ]
        self.problem_files_path = problem_files_path
        self.file_column = file_column
        self.pixel_size_um = pixel_size_um
        self.convert_coordinates = convert_coordinates
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.df: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_trimmed: Optional[pd.DataFrame] = None

        self.stats_results: Optional[Dict[str, pd.DataFrame]] = None
        self.regression_results: Optional[Dict[str, pd.DataFrame]] = None
        self.random_forest_results: Optional[pd.DataFrame] = None

        self.default_cell_type = default_cell_type
        self.default_treatment = default_treatment
        self.label_map_path = label_map_path

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
        df = _consolidate_filenames(df, prefer="FileName_Act", output_col="FileName")

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
        df = add_aspect_ratio(df, prefixes=["nuc", "act"], overwrite=False)

        # per-group robust outlier flags
        qc_cols = [
            "nuc_AreaShape_Area",
            "act_AreaShape_Area",
            # choose your best nuclear intensity feature; adjust if your column differs:
            "nuc_Intensity_MeanIntensity",
        ]
        existing_qc_cols = [c for c in qc_cols if c in df.columns]

        df = flag_outliers_per_group(
            df,
            group_cols=[c for c in ["cell_type", "treatment"] if c in df.columns],
            cols_mad_log=existing_qc_cols,
            z_cutoff=3.5,
            mitotic_rule=True,
            mitotic_intensity_col=None,  # auto-detect nuc intensity if None
            mitotic_area_col=None,  # defaults to act area if None
        )

        # final assignments
        self.df = df.reset_index(drop=True)  # the cleaned dataset used by stats etc.
        self.df_clean = self.df.copy()  # a pristine copy for re-analysis

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
        df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
        if df is None:
            raise RuntimeError("No data available. Did you run load_and_clean()?")

        # --- choose numeric feature columns (drop metadata) ---
        meta_cols = {
            "ImageNumber",
            "ObjectNumber",
            "FileName",
            "group",
            "cell_type",
            "treatment",
        }
        num_df = df.select_dtypes(include=[np.number]).copy()
        # drop any metadata that slipped in
        num_df = num_df.drop(
            columns=[c for c in meta_cols if c in num_df.columns], errors="ignore"
        )
        if num_df.empty:
            self.stats_results = {
                "note": pd.DataFrame([{"msg": "No numeric features"}])
            }
            return

        # --- helper stats ---
        def sem(x):
            n = x.count()
            return (x.std(ddof=1) / np.sqrt(n)) if n > 0 else np.nan

        def ci_low(x):
            n = x.count()
            s = x.std(ddof=1)
            m = x.mean()
            return (m - 1.96 * s / np.sqrt(n)) if n > 1 and np.isfinite(s) else np.nan

        def ci_high(x):
            n = x.count()
            s = x.std(ddof=1)
            m = x.mean()
            return (m + 1.96 * s / np.sqrt(n)) if n > 1 and np.isfinite(s) else np.nan

        def iqr(x):
            x = x.dropna()
            return np.nan if x.empty else (np.percentile(x, 75) - np.percentile(x, 25))

        agg_funcs = ["count", "mean", "std", sem, ci_low, ci_high, "median", iqr]
        stat_names = [
            "count",
            "mean",
            "std",
            "sem",
            "ci95_low",
            "ci95_high",
            "median",
            "iqr",
        ]

        # --- overall (all rows) ---
        overall = num_df.agg(agg_funcs).T
        overall.columns = stat_names

        results = {"overall": overall}

        # --- by group (e.g., HeLa_ctl, HeLa_ble, ...) ---
        if "group" in df.columns:
            g = df.groupby("group", observed=False)[num_df.columns].agg(
                agg_funcs
            )  # MultiIndex cols: (feature, stat)

            # flatten to f"{feature}__{stat}"
            g_flat = g.copy()
            g_flat.columns = [
                f"{feat}__{(fn if isinstance(fn, str) else fn.__name__)}"
                for feat, fn in g_flat.columns.to_list()
            ]
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


#    def run_regression(self) -> None:
#         """Toy least-squares regression: first numeric col as y, rest as X."""
#         df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
#         if df is None:
#             raise RuntimeError("No data available. Did you run load_and_clean()?")

#         numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
#         if len(numeric_cols) < 2:
#             self.regression_results = {"meta": pd.DataFrame([{"note": "Insufficient numeric features for regression."}])}
#             return

#         y_col = numeric_cols[0]
#         X_cols = numeric_cols[1:]
#         y = df[y_col].astype(float).values
#         X = df[X_cols].astype(float).values
#         X_design = np.column_stack([np.ones(len(X)), X])

#         beta, residuals, rank, s = np.linalg.lstsq(X_design, y, rcond=None)
#         y_hat = X_design @ beta
#         ss_tot = float(np.sum((y - y.mean()) ** 2))
#         ss_res = float(np.sum((y - y_hat) ** 2))
#         r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

#         coef_df = pd.DataFrame({"term": ["intercept"] + X_cols, "coef": beta})
#         metrics_df = pd.DataFrame([{"target": y_col, "r2": r2, "n": len(y)}])

#         self.regression_results = {"coefficients": coef_df, "metrics": metrics_df}

#      def run_random_forest(self) -> None:
#         """Correlation-based 'importance' proxy (no sklearn dep)."""
#         df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
#         if df is None:
#             raise RuntimeError("No data available. Did you run load_and_clean()?")

#         numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
#         if len(numeric_cols) < 2:
#             self.random_forest_results = pd.DataFrame([{"note": "Insufficient numeric features for importance."}])
#             return

#         y_col = numeric_cols[0]
#         X_cols = numeric_cols[1:]
#         y = df[y_col].astype(float)
#         rows: List[Dict[str, float]] = []
#         for c in X_cols:
#             x = df[c].astype(float)
#             if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
#                 corr = 0.0
#             else:
#                 corr = float(np.corrcoef(x, y)[0, 1])
#             rows.append({"feature": c, "abs_correlation": abs(corr)})
#         self.random_forest_results = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


# -------------------------- utilities --------------------------


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
