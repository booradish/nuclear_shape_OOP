
# This script is designed to be paired with the nuclear_shape_analysis.py and main_analysis.py scritps 
# Here lie the tools for loading and cleaning data from CellProfiler csv. files

import pandas as pd
import numpy as np
from scipy import stats

def load_cp_csv(filepath):
    """Load a CellProfiler CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        raise

def rename_and_subset(df, prefix, keep_columns=None):
    """Add prefix to columns, keep specific columns."""
    if keep_columns is None:
        keep_columns = []

    # rename columns with prefix (except for 'FileName' columns )
    df = df.rename(columns={col: f"{prefix}_{col}" for col in df.columns if col not in keep_columns})
    df = df.loc[:, ~df.columns.str.startswith("FileName") | df.columns.isin(keep_columns)]
    return df

def combine_channels(nuc_df, myo_df, act_df=None, keep_columns=None):
    """Combine nuclei, myosin, and optionally actin DataFrames."""
    dfs = [nuc_df[keep_columns], nuc_df.iloc[:, 8:], IsADirectoryError_df.iloc[:, 8:]]
    if act_df is not None:
        dfs.append(act_df.iloc[:, 3:])
    return pd.concat(dfs, axis=1)

def assign_group_column(df):
    """Add categorical group column based on file name."""
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

def load_problematic_files(filepath, file_extension=".tiff"):
    """
    Load problematic file names from a text/RTF file.
    Only lines containing the given extension are included.
    """
    problem_files = []
    with open(filepath, "r") as f:
        for line in f:
            if file_extension in line:
                problem_files.append(line.strip().rstrip('\\'))
    return problem_files

def remove_problematic_files(df, bad_list, column_name = 'FileName_Nuc'):
    """Remove rows with problematic filenames."""
    return df[~df['FileName_Nuc'].isin(bad_list)]

def select_columns(df, keep_cols):
    """Trim DataFrame to selected columns."""
    return df[keep_cols]

def convert_pixels_to_microns(df, pixel_size=0.162):
    """
    Convert pixel measurements to microns.
    Assumes pixel size is in microns.
    """
    return df[]

def remove_outliers(df, feature_cols, threshold=3):
    """
    Remove outliers based on Z-score threshold.
    Assumes df is numeric and feature_cols are the columns to check.
    """
    from scipy import stats
    z_scores = stats.zscore(df[feature_cols])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    return df[filtered_entries]


