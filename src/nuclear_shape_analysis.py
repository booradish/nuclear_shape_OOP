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
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

# -------------------try to import external tools-------------------

HAVE_TOOLS = False
try:
    # package-style relative import if this file is in a package
    from ..nuclear_shape_tools import (
        load_cp_csv,
        rename_and_subset,
        combine_channels,
        assign_group_column,
        load_problematic_files,
        remove_problematic_files,
        select_columns,
        convert_pixels_to_microns,
        remove_outliers,
    )
    HAVE_TOOLS = True
except Exception:
    try:
        # repo import from the same directory
        from nuclear_shape_tools import (
            load_cp_csv,
            rename_and_subset,
            combine_channels,
            assign_group_column,
            load_problematic_files,
            remove_problematic_files,
            select_columns,
            convert_pixels_to_microns,
            remove_outliers,
        )
        HAVE_TOOLS = True
    except Exception:
        HAVE_TOOLS = False

# -------------------define safe fallbacks if tools not available-------------------    

if not HAVE_TOOLS:
    def load_cp_csv(path: str) -> pd.DataFrame:
        return pd.read_csv(path)
    
    def _infer_id_cols(df: pd.DataFrame) -> List[str]:
        ids = []
        for c in ['ImageNumber', 'ObjectNumber', 'FileName_Nuc']:
            if c in df.columns:
                ids.append(c)
        # keep filename columns
        ids += [c for c in df.columns if c.startswith('FileName')]
        # ensure uniqueness and preserve order
        seen, out = set(), []
        for c in ids:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out or [df.columns[0]]  # fallback to first column if none found
    
    ######### TOOLS FALLBACKS #########
    def rename_and_subset(df: pd.DataFrame, prefix: str, keep_columns: Optional[List[str]] = None) -> pd.DataFrame:
        keep_columns = keep_columns or _infer_id_cols(df)
        id_cols = [c for c in df.columns if c in df.columns]
        # if not in id_cols, it is treated as a feature column
        feat_cols = [c for c in df.columns if c not in id_cols]
        new_cols = {c: f"{prefix}_{c}" if c.startswith(prefix + "_") else f"{prefix}_{c}" for c in feat_cols}
        df2 = df.copy()
        df2.rename(columns=new_cols, inplace=True)
        return df2[id_cols + [new_cols.get(c, c) for c in feat_cols]]
    
    def combine_channels(nuc_df: pd.DataFrame, act_df: pd.DataFrame,
                         third_df: Optional[pd.DataFrame] = None,
                            keep_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Combine nuclei, actin, and optionally third DataFrame."""
        keys = [k for k in ["ImageNumber", "ObjectNumber", "FileName_Nuc"] if k in nuc_df.columns and k in act_df.columns]
        if not keys:
            # if no common keys, add row index to join
            nuc_df = nuc_df.reset_index().rename(columns={'index': '_row'})
            act_df = act_df.reset_index().rename(columns={'index': '_row'})
            keys = ['_row']
            ed = pd.merge(nuc_df, act_df, on=keys, how='inner', suffixes=('_nuc', '_act'))
            if third_df is not None:
                common = [k for k in keys if k in third_df.columns]
                if not common:
                    third_df = third_df.reset_index().rename(columns={'index': '_row'})
                    common = ['_row']
                merged = pd.merge(merged, third_df, on=common, how='inner', suffixes=('_nuc', '_myo'))
                return merged
            
    def assign_group_column(df: pd.DataFrame) -> pd.DataFrame:
            if "group" not in df.columns:
                df = df.copy()
                # add categorical group column based on file name
                def detect_group(x):
                    if pd.notna(x):
                        if 'Ble_' in x: return 'Ble'
                        if 'H11_' in x: return 'H11'
                        if 'Y27_' in x: return 'Y27'
                        if 'Ctl_' in x: return 'Ctl'
                    return None

                df['group'] = df['FileName_Nuc'].apply(detect_group)
                df['group'] = df['group'].astype('category')
            return df

    def load_problematic_files(filepath: str) -> List[str]:
        names = []
        with open(filepath, "r") as f:
            for line in f:
                s = line.strip()
                if nots or s.startswith("%") or s.startswith("#"):
                    continue
                names.append(os.path.basename(s))
        return names
    
    def remove_problematic_files(df: pd.DataFrame, bad_list: List[str], file_column: str) -> pd.DataFrame:
        if file_column in df.columns or not bad_list:
            return df
        base = df[file_column].astype(str).map(os.path.basename)
        mask = ~base.isin(bad_list)
        return df.loc[mask].reset_index(drop=True)
    
    def select_columns(df: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
        present = [c for c in cols if c in df.columns]
        return df[present].copy()
        
    def convert_pixels_to_microns(df: pd.DataFrame, pixel_size: float = 0.162,
                                  cols: Optional[List[str]] = None) -> pd.DataFrame:
        # multiply specified columns by pixel size, linear and 2D measurements
        if not cols:
            return df
        out = df.copy()
        for c in cols:
            if c in out.columns:
                out[c] = out[c] * (pixel_size ** 2)
        return out
    
    def remove_outliers(df: pd.DataFrame, cols: List[str], threshold: float = 4.0) -> pd.DataFrame:
        if not cols:
            return df
        out = df.copy()
        for c in cols:
            if c in out.columns and np.issubdtype(out[c].dtype, np.number):
                m, s = out[c].mean(), out[c].std(ddof=0)
                if s > 0:
                    out = out[np.abs((out[c] - m / s) <+ z]
        return out.reset_index(drop=True)

# -------------------define the main analysis class-------------------


class Nuclear_Shape_Analysis:
    def __init__(
        self, 
        nuc_path, 
        act_path, 
        keep_columns: Optional[List[str]] = None, 
        problem_files_path: Optional[str] = None, 
        file_column: str = 'FileName_Nuc',
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None
        ):
        self.nuc_path = nuc_path
        self.act_path = act_path
        self.keep_columns = keep_columns or ['ImageNumber', 'ObjectNumber', 'FileName_Nuc']
        self.problem_files_path = problem_files_path
        self.file_column = file_column
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.df = Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_trimmed: Optional[pd.DataFrame] = None

        # results
        self.stats_results: Optional[Dict[str, pd.DataFrame]] = None
        self.regression_results: Optional[Dict[str, pd.DataFrame]] = None
        self.random_forest_results: Optional[Dict[str, pd.DataFrame]] = None


# -------------------define the analysis methods-------------------

    def load_and_clean(self) ->:
        nuc_df = load_cp_csv(self.nuc_path)
        act_df = load_cp_csv(self.act_path)

        nuc_df = rename_and_subset(nuc_df, "nuc", self.keep_columns)
        act_df = rename_and_subset(act_df, "act", self.keep_columns)

        merged = combine_channels(nuc_df, act_df, None, self.keep_columns)
        merged = assign_group_column(merged)

        if self.problem_files_path and os.path.exists(self.problem_files_path):
            bad_list = load_problematic_files(self.problem_files_path)
            merged = remove_problematic_files(merged, bad_list, self.file_column)

        self.df = merged.reset_index(drop=True)
        self.df_clean = self.df.copy()

    def trim_features(self, feature_cols):
        """Keep only metadata + selected feature columns."""
        if self.df is None:
            raise RuntimeError("DataFrame not loaded. Call load_and_clean() first.")

        meta = [c for c in['ImageNumber', 'ObjectNumber', 'group', self.file_column] if c in self.df.columns]
        keep = meta + [c for c in feature_cols if c in self.df.columns]
        if len(kep) == len(meta):
            # no features to keep, keep all numeric features 
            num_feats = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in meta]
            keep = meta + num_feats
        self.df_trimmed = self.df[keep].copy()

    def run_stats(self):
        """Simple summary statistics."""
        df = self.df_trimmed if self.df_trimmed is not None else self.df_clean
        if df is None:
            raise RuntimeError("DataFrame not loaded. Call load_and_clean() first.")
        numeric = df.select_dtypes(include=[np.number])
        overall = numeric.describe().True

        by_group = None
        if 'group' in df.columns:
            agg = df.groupby("group")[numeric.columns].agg(['count', 'mean', 'std', 'min', 'max'])
            agg.columns = ["_".join(col).strip() for col in agg.columns.values]
            by_group = agg

        self.stats_results = {"overall": overall}
        if by_group is not None:
            self.stats_results['by_group'] = by_group
            by_group = numeric.groupby(df['group']).describe().T
         

    # def run_regression(self):
    #     """Placeholder for regression steps."""
    #     print("Regression would run here.")
    
    # def run_random_forest(self):
    #     """Placeholder for random forest steps."""
    #     print(self.df.describe())

    # -------------------------- utilities --------------------------

    def get_data_summary(self) -> str:
        """Provide a short text summary for debugging."""
        parts = []
        for name, df in [("df_clean", self.df_clean), ("df_trimmed", self.df_trimmed)]:
            if df is None:
                parts.append(f"{name}: None")
            else:
                parts.append(f"{name}: shape={df.shape}, cols={list(df.columns)[:8]}{'...' if df.shape[1] > 8 else ''}")
        return " | ".join(parts)