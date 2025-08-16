# main analysis script

# This script is designed to be paired with the nuclear_shape_analysis.py and main_shape_tools.py scritps
# Here lie the tools for cleaning the data and running the main analysis pipeline on data from CellProfiler csv. files

# ------------Import libraries --------------------------------
import argparse
import os  # For file path operations
import json
import datetime
import pandas as pd  # For data manipulation and analysis

# ------- Import custom modules ------------------------------------------------------------

# Import the main analysis class from GitHub repository or local src folder
# Import from src folder (if you have this structure)
try:
    from src.nuclear_shape_analysis import Nuclear_Shape_Analysis
except ImportError:
    try:
        from nuclear_shape_analysis import Nuclear_Shape_Analysis
    except ImportError:
        print("✗ Could not import Nuclear_Shape_Analysis from src/ or local directory.")
        raise


def parse_csv_list(s: str | None):
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def check_file(path: str, label: str) -> bool:
    if not path or not os.path.exists(path):
        print(f"✗ {label} file NOT found: {path}")
        return False
    print(f"✓ {label} file found: {path}")
    return True


def build_parser():
    p = argparse.ArgumentParser("Run nuclear–cell analysis pipeline")
    p.add_argument("--nuc_path", required=True)
    p.add_argument("--act_path", required=True)
    p.add_argument("--problem_files_path", default=None)
    p.add_argument(
        "--keep_columns", default="ImageNumber,ObjectNumber,FileName_Nuc,FileName_Act"
    )
    p.add_argument("--features", default="")
    p.add_argument("--file_column", default="FileName_Act")
    p.add_argument("--pixel_size_um", type=float, default=None)
    # optional: labeling overrides
    p.add_argument("--cell_type", default=None)
    p.add_argument("--treatment", default=None)
    p.add_argument("--label_map", default=None)
    p.add_argument(
        "--output_dir", default="outputs/run", help="Folder to write CSV outputs"
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print("=== CHECKING FILE PATHS ===")
    ok = check_file(args.nuc_path, "Nuclear") & check_file(args.act_path, "Actin")
    prob = args.problem_files_path
    if prob and not os.path.exists(prob):
        print(f"⚠ Problem files NOT found, continuing without: {prob}")
        prob = None
    if not ok:
        print("✗ Aborting due to missing inputs.")
        return 2

    keep_columns = parse_csv_list(args.keep_columns)
    features = parse_csv_list(args.features)

    print("\n=== Initializing Analysis ===")
    cp = Nuclear_Shape_Analysis(
        nuc_path=args.nuc_path,
        act_path=args.act_path,
        keep_columns=keep_columns,
        problem_files_path=prob,
        file_column=args.file_column,
        pixel_size_um=args.pixel_size_um,
        default_cell_type=args.cell_type,
        default_treatment=args.treatment,
        label_map_path=args.label_map,
    )
    print("✓ Analysis class initialized successfully.")

    print("\n=== Running Analysis Pipeline ===")
    cp.load_and_clean()
    print("✓ Data loaded and cleaned successfully.")

    if features:
        cp.trim_features(features)
        print("✓ Features trimmed successfully.")
    else:
        print("• No --features provided; keeping all numeric features by default.")
        cp.trim_features([])

    cp.run_stats()
    print("✓ Stats done.")
    # cp.run_regression();   print("✓ Regression done.")
    # cp.run_random_forest();print("✓ Importance done.")

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    if cp.stats_results:
        for name, tbl in cp.stats_results.items():
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                tbl.to_csv(os.path.join(outdir, f"stats_{name}.csv"))

    # save dataframes to CSV
    if cp.df_clean is not None:
        cp.df_clean.to_csv(os.path.join(outdir, "clean_merged_df.csv"), index=False)
    if cp.df_trimmed is not None:
        cp.df_trimmed.to_csv(
            os.path.join(outdir, "trimmed_features_df.csv"), index=False
        )

    # save stats
    if cp.stats_results:
        for name, tbl in cp.stats_results.items():
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                tbl.to_csv(os.path.join(args.output_dir, f"stats_{name}.csv"))

    # save regression results
    if cp.regression_results is not None:
        for name, df in cp.regression_results.items():
            df.to_csv(os.path.join(outdir, f"regression_{name}.csv"), index=False)

    # save random forest results
    if cp.random_forest_results is not None:
        cp.random_forest_results.to_csv(
            os.path.join(outdir, "random_forest_importance.csv"), index=False
        )

    # save run metadata
    meta = {
        "nuc_path": args.nuc_path,
        "act_path": args.act_path,
        "pixel_size_um": args.pixel_size_um,
        "cell_type": args.cell_type,
        "treatment": args.treatment,
        "timestamp": datetime.datetime.now().isoformat(),
        "n_rows_clean": int(cp.df_clean.shape[0]) if cp.df_clean is not None else 0,
        "n_cols_clean": int(cp.df_clean.shape[1]) if cp.df_clean is not None else 0,
    }
    with open(os.path.join(outdir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✓ Pipeline finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
