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
    # Which steps to run (comma list): stats,regress,importance,compare
    p.add_argument(
        "--run",
        default="stats,importance",  # sensible default; add 'compare' when you want it
        help="Comma-separated steps to run: stats,regress,importance,compare",
    )

    # Quality control options
    p.add_argument(
        "--qc_mode",
        choices=["none", "robust"],
        default="robust",
        help="Outlier handling: none (no flagging) or robust (MAD+mitotic rule)",
    )
    p.add_argument(
        "--qc_drop",
        action="store_true",
        help="If set, drop rows flagged by QC before analysis (default: drop when robust).",
    )
    p.add_argument(
        "--qc_keep",
        dest="qc_drop",
        action="store_false",
        help="Keep flagged rows (only mark qc_keep/qc_reason).",
    )
    p.set_defaults(qc_drop=True)

    # Feature engineering toggles
    p.add_argument(
        "--no_aspect_ratio",
        action="store_true",
        help="Do not add nuc/act aspect ratio derived feature.",
    )
    p.add_argument(
        "--no_convert_coords",
        action="store_true",
        help="Do not convert coordinate-like columns to microns.",
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

    steps = set(s.strip() for s in args.run.split(",") if s.strip())

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
        qc_mode=args.qc_mode,
        qc_drop=args.qc_drop,
        add_aspect_ratio=not args.no_aspect_ratio,
    )
    print("✓ Analysis class initialized successfully.")

    print("\n=== Running Analysis Pipeline ===")

    cp.load_and_clean()
    print("✓ Data loaded and cleaned successfully.")

    # Print a quick QC summary to the console
    removed = cp.n_rows_before - cp.n_rows_after
    pct = (removed / cp.n_rows_before * 100.0) if cp.n_rows_before else 0.0
    print(
        f"QC: {cp.n_rows_before:,} rows → {cp.n_rows_after:,} rows "
        f"(removed {removed:,}, {pct:.2f}%)."
    )

    if getattr(cp, "qc_summary", None) is not None:
        print("\nQC per group:")
        print(cp.qc_summary.to_string(index=False))

    # Save to output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    if getattr(cp, "qc_summary", None) is not None:
        cp.qc_summary.to_csv(
            os.path.join(args.output_dir, "qc_summary.csv"), index=False
        )

    # Optional: save the flagged dataset before dropping outliers
    if getattr(cp, "df_flags", None) is not None:
        cp.df_flags.to_csv(
            os.path.join(args.output_dir, "cleaned_with_flags.csv"), index=False
        )

    if features:
        cp.trim_features(features)
        print("✓ Features trimmed successfully.")
    else:
        print("• No --features provided; keeping all numeric features by default.")
        cp.trim_features([])

    if "stats" in steps:
        cp.run_stats()
        print("✓ Stats done.")
    if "regress" in steps:
        cp.run_regression()
        print("✓ Regression done.")
    if "importance" in steps:
        cp.run_random_forest()
        print("✓ Importance done.")
    if "compare" in steps:
        comp = cp.compare_vs_control(
            control_label="ctl", group_col="treatment", stratify_by="cell_type"
        )
        if not comp.empty:
            comp.to_csv(
                os.path.join(args.output_dir, "compare_vs_ctl.csv"), index=False
            )
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
