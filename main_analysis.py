# main analysis script

# This script is designed to be paired with the nuclear_shape_analysis.py and main_shape_tools.py scritps
# Here lie the tools for cleaning the data and running the main analysis pipeline on data from CellProfiler csv. files

# ------------Import libraries --------------------------------
import argparse
import os  # For file path operations
import json
import datetime
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations

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
    p.add_argument("--keep_columns",
                   default="ImageNumber,ObjectNumber,FileName_Nuc,FileName_Act"
    )
    p.add_argument("--features", default="")
    p.add_argument("--file_column", default="FileName_Act")
    p.add_argument("--pixel_size_um", type=float, default=None)
    # optional: labeling overrides
    p.add_argument("--cell_type", default=None)
    p.add_argument("--treatment", default=None)
    p.add_argument("--label_map", default=None)
    p.add_argument("--normalize_mode",
        choices=["none", "zscore", "robust"],
        default="none",
        help="Add normalized copies of feature columns.",
    )
    p.add_argument("--normalize_by_group",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, normalize within cell_type×treatment groups.",
    )
    p.add_argument("--use_norm_for_stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, run_stats uses the normalized copies.",
    )
    p.add_argument("--norm_suffix",
        default="_z",
        help="Suffix to append to normalized columns (default: _z).",
    )
    p.add_argument("--output_dir", default="outputs/run", help="Folder to write CSV outputs"
    )
    # Which steps to run (comma list): stats,regress,importance,compare
    p.add_argument("--run",
        default="stats,importance",  # sensible default; add 'compare' when you want it
        help="Comma-separated steps to run: stats,regress,importance,compare",
    )

    # Quality control options
    p.add_argument("--qc_mode",
        choices=["none", "robust"],
        default="robust",
        help="Outlier handling: none (no flagging) or robust (MAD+mitotic rule)",
    )
    p.add_argument("--qc_drop",
        action="store_true",
        help="If set, drop rows flagged by QC before analysis (default: drop when robust).",
    )
    p.add_argument("--qc_keep",
        dest="qc_drop",
        action="store_false",
        help="Keep flagged rows (only mark qc_keep/qc_reason).",
    )
    p.set_defaults(qc_drop=True)

    # Feature engineering toggles
    p.add_argument("--no_aspect_ratio",
        action="store_true",
        help="Do not add nuc/act aspect ratio derived feature.",
    )
    p.add_argument("--no_convert_coords",
        action="store_true",
        help="Do not convert coordinate-like columns to microns.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    print("=== CHECKING FILE PATHS ===")
    ok = check_file(args.nuc_path, "Nuclei") & check_file(args.act_path, "Cells")
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

    cp: Nuclear_Shape_Analysis = Nuclear_Shape_Analysis(
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
        normalize_mode=args.normalize_mode,
        normalize_by_group=args.normalize_by_group,
        use_norm_for_stats=args.use_norm_for_stats,
        norm_suffix=args.norm_suffix,
    )
    print("✓ Analysis class initialized successfully.")

    print("\n=== Running Analysis Pipeline ===")
    cp.load_and_clean()
    print("✓ Data loaded and cleaned successfully.")

    # --- QC summary & save ---
    # Make sure the output directory exists before writing files
    os.makedirs(args.output_dir, exist_ok=True)

    removed = cp.n_rows_before - cp.n_rows_after
    pct = (removed / cp.n_rows_before * 100.0) if cp.n_rows_before else 0.0
    print(
        f"QC: {cp.n_rows_before:,} rows → {cp.n_rows_after:,} rows (removed {removed:,}, {pct:.2f}%)."
    )

    qs = getattr(cp, "qc_summary", None)
    if isinstance(qs, pd.DataFrame) and not qs.empty:
        print("\nQC per group:")
        print(qs.to_string(index=False))
        qs.to_csv(os.path.join(args.output_dir, "qc_summary.csv"), index=False)

    flags = getattr(cp, "df_flags", None)
    if isinstance(flags, pd.DataFrame) and not flags.empty:
        flags.to_csv(
            os.path.join(args.output_dir, "cleaned_with_flags.csv"), index=False
        )

    # Trim (keeps group/cell_type/treatment; you already updated trim_features)
    if features:
        cp.trim_features(features)
        print("✓ Features trimmed successfully.")
    else:
        print("• No --features provided; keeping all numeric features by default.")
        cp.trim_features([])

    outdir = args.output_dir

    # === Stats =========================================================
    if "stats" in steps:
        cp.run_stats()
        print("✓ Stats done.")
        if cp.stats_results:
            for name, tbl in cp.stats_results.items():
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    tbl.to_csv(os.path.join(outdir, f"stats_{name}.csv"), index=False)

    # === Determine present cell types ============================
    present_ct = None
    dfc = getattr(cp, "df_clean", None)
    if isinstance(dfc, pd.DataFrame) and "cell_type" in dfc.columns:
        present_ct = (
            sorted(
                x
                for x in dfc["cell_type"].astype(str).unique()
                if x and x.lower() != "unknown"
            )
            or None
        )  # leave None if no valid types

    # === PCA on cell/act features ================================
    if "pca" in steps:
        cp.run_pca(predictors="act", use_norm=True, filter_cell_types=present_ct)
        print("✓ PCA done.")
        # Save PCA artifacts
        pr = getattr(cp, "pca_results", None)
        if pr:
            # explained variance table
            ev = pd.DataFrame(
                {
                    "component": np.arange(
                        1, len(pr["explained_variance_ratio"]) + 1, dtype=int
                    ),
                    "explained_variance": pr["explained_variance"],
                    "explained_variance_ratio": pr["explained_variance_ratio"],
                    "cumulative_variance_ratio": np.cumsum(
                        pr["explained_variance_ratio"]
                    ),
                }
            )
            ev.to_csv(os.path.join(outdir, "pca_explained_variance.csv"), index=False)
            # loadings (features x PCs)
            loadings = pd.DataFrame(pr["Vt"].T, index=pr["X_cols"])
            loadings.columns = [f"PC{i}" for i in range(1, loadings.shape[1] + 1)]
            loadings.to_csv(os.path.join(outdir, "pca_loadings.csv"))

    # === Regress nuc features on top PCs(act) ====================
    if "pca_reg" in steps:
        cp.run_regression(
            targets="nuc",
            predictors="act",
            var_threshold=0.95,  # or set n_components=
            use_norm=True,
            filter_cell_types=present_ct,
        )
        print("✓ PCA regression done.")
        # Save PCA regression results (nested dict)
        if cp.regression_results and "pca" in cp.regression_results:
            pca_res = cp.regression_results["pca"]
            coef = pca_res.get("coefficients")
            mets = pca_res.get("metrics")
            if isinstance(coef, pd.DataFrame):
                coef.to_csv(
                    os.path.join(outdir, "regression_pca_coefficients.csv"), index=False
                )
            if isinstance(mets, pd.DataFrame):
                mets.to_csv(
                    os.path.join(outdir, "regression_pca_metrics.csv"), index=False
                )

    # ==== Toy regression / importance ========================
    if "regress" in steps:
        cp.run_regression()
        print("✓ Regression done.")

        reg = getattr(cp, "regression_results", None)
        if isinstance(reg, dict):
            for name, df in reg.items():
                if isinstance(df, pd.DataFrame):
                    df.to_csv(
                        os.path.join(outdir, f"regression_{name}.csv"), index=False
                    )
        elif isinstance(reg, pd.DataFrame):
            # (just in case a future version returns a single DF)
            reg.to_csv(os.path.join(outdir, "regression_results.csv"), index=False)

    if "importance" in steps:
        cp.run_random_forest()
        print("✓ Importance done.")
        if cp.random_forest_results is not None:
            cp.random_forest_results.to_csv(
                os.path.join(outdir, "random_forest_importance.csv"), index=False
            )

    # === Compare vs control ===========================================
    if "compare" in steps:
        comp = cp.compare_vs_control(
            control_label="ctl", group_col="treatment", stratify_by="cell_type"
        )
        if isinstance(comp, pd.DataFrame) and not comp.empty:
            comp.to_csv(os.path.join(outdir, "compare_vs_ctl.csv"), index=False)

    # === Make plots ====================================================
    cp.output_dir = args.output_dir  # expose to class (figure defaults)
    pretty_map = {
        "nuc_AreaShape_Area": "Nuclear Area (µm²)",
        "nuc_AreaShape_Perimeter": "Nuclear Perimeter (µm)",
        "nuc_AreaShape_MajorAxisLength": "Nuclear Major Axis (µm)",
        "nuc_AreaShape_MinorAxisLength": "Nuclear Minor Axis (µm)",
        "nuc_AreaShape_AspectRatio": "Nuclear Aspect Ratio",
        "nuc_AreaShape_Circularity": "Nuclear Circularity",
        "nuc_AreaShape_Eccentricity": "Nuclear Eccentricity",
        "nuc_AreaShape_Solidity": "Nuclear Solidity",
        "act_AreaShape_Area": "Cell Area (µm²)",
        "act_AreaShape_Perimeter": "Cell Perimeter (µm)",
        "act_AreaShape_MajorAxisLength": "Cell Major Axis (µm)",
        "act_AreaShape_MinorAxisLength": "Cell Minor Axis (µm)",
        "act_AreaShape_AspectRatio": "Cell Aspect Ratio",
        "act_AreaShape_Circularity": "Cell Circularity",
        "act_AreaShape_Eccentricity": "Cell Eccentricity",
        "act_AreaShape_Solidity": "Cell Solidity",
        "act_AreaShape_Compactness": "Cell Compactness",
    }
    plot_cols = list(pretty_map.keys())
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plots_summary = cp.plot_by_celltype(
        columns=plot_cols,
        pretty_names=pretty_map,
        group_col="treatment",
        use_norm=False,  # set True if you want normalized copies in plots
        save_dir=fig_dir,
        n_bootstrap=2000,
        random_state=42,
    )
    plots_summary.to_csv(
        os.path.join(fig_dir, "plots_summary_all_celltypes.csv"), index=False
    )
    print(f"✓ Plots saved to {fig_dir}")

    # === Save core tables ==============================================
    if cp.df_clean is not None:
        cp.df_clean.to_csv(os.path.join(outdir, "clean_merged_df.csv"), index=False)
    if cp.df_trimmed is not None:
        cp.df_trimmed.to_csv(
            os.path.join(outdir, "trimmed_features_df.csv"), index=False
        )

    # === Save run metadata =============================================
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
