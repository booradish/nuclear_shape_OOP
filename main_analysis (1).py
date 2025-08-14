
# main analysis script

# This script is designed to be paired with the nuclear_shape_analysis.py and main_shape_tools.py scritps 
# Here lie the tools for cleaning the data and running the main analysis pipeline on data from CellProfiler csv. files 

# === import libraries =========================================================
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import os  # For file path operations
from scipy import stats  # For statistical operations
import sys  # For system-specific parameters and functions

# === Import classes ===============================================================

# Import the main analysis class from GitHub repository or local src folder
# Import from src folder (if you have this structure)
try:
    from src.nuclear_shape_analysis import Nuclear_Shape_Analysis
except ImportError:
    # If src folder doesn't exist, import directly
    from nuclear_shape_OOP.src.nuclear_shape_analysis import Nuclear_Shape_Analysis

# === get data ================================================================

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define the file path
file_path = "/Users/rosi_dakd/My Drive/Prasad_Lab/nuclear_shape/Data/20240815_HeLa_TMI_40X_output/2D_CellProfiler_output/"

# Load the CSV files using the full path
nuc_path = (file_path + "2D_3Channel_Output_FilteredNucleiMyo.csv")
act_path = (file_path + "2D_3channel_Output_cellsAct.csv")

problem_files_path = "/Users/rosi_dakd/My Drive/Prasad_Lab/nuclear_shape/Data/20240815_HeLa_TMI_40X_output//Users/rosi_dakd/My Drive (rdanzman@gmail.com)/Prasad_Lab/nuclear_shape/Data/20240815_HeLa_TMI_40X_output/2D_CellProfiler_output/Output_Segmentation_Images/20240815_HeLa_TMI_40X_MAX_Segmented_problemImages.tex"

keep_columns = ['ImageNumber', 'ObjectNumber', 'FileName_Nuc']

# === Check if files exist ========================================================

print("=== CHECKING FILE PATHS ===")
for name, path in [("Nuclear", nuc_path), ("Actin", act_path)]:
    if os.path.exists(path):
        print(f"✓ {name} file found: {path}")
    else:
        print(f"✗ {name} file NOT found: {path}")

if os.path.exists(problem_files_path):
    print(f"✓ Problem files found: {problem_files_path}")
else:
    print(f"✗ Problem files NOT found: {problem_files_path}")
    problem_files_path = None  # Set to None if file doesn't exist

# === Instantiate the analysis class ===============================================================

print("\n=== Initializing Analysis ===")
try:
    cp = Nuclear_Shape_Analysis(
        nuc_path=nuc_path,
        act_path=act_path,
        keep_columns=keep_columns,
        problem_files_path=problem_files_path,
        file_column='FileName_Act'
    )
    print("✓ Analysis class initialized successfully.")
except Exception as e:
    print(f"✗ Error initializing analysis class: {e}")
    sys.exit(1)  # Exit if initialization fails

# === Run pipeline ===============================================================

print("\n=== Running Analysis Pipeline ===")
try:
    # Load and clean data
    cp.load_and_clean()
    print("✓ Data loaded and cleaned successfully.")

    # Trim features to keep only specified columns
    cp.trim_features(['nuc_Area', 'nuc_AspectRatio', 'myo_Intensity_MeanIntensity'])
    print("✓ Features trimmed successfully.")

    # Run statistical analysis
    cp.run_stats()
    print("✓ Statistical analysis completed successfully.")

    # Run regression analysis
    cp.run_regression()
    print("✓ Regression analysis completed successfully.")

    # Run random forest analysis
    cp.run_random_forest()
    print("✓ Random forest analysis completed successfully.")

except Exception as e:
    print(f"✗ Error during analysis pipeline: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()  # Print the full traceback for debugging

    try:
        summary = cp.get_data_summary()
        print(f"\nDebugging info - Current data summary: {summary}")
    except Exception as e:
        print(f"✗ Error getting data summary: {e}")
        sys.exit(1)


