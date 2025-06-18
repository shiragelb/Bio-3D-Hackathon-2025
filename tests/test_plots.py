import pandas as pd
from plotting_utils import (
    preprocess_pipeline_output,
    plot_score_distribution,
    plot_roc_curve,
    plot_score_boxplot
)
import os

def run_tests():
    """
    Main function to run all tests for the plotting utilities.
    """
    # Define the path to the dummy data file
    input_data_path = '../DB/dummy2.csv'
    # Define the directory where plots will be saved
    output_dir = "output_plots"

    # --- Step 1: Check if dummy data file exists ---
    if not os.path.exists(input_data_path):
        print(f"Error: Dummy data file not found at '{input_data_path}'")
        print("Please make sure you have created the dummy CSV file.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to the '{output_dir}/' directory.")

    print(f"Loading dummy data from '{input_data_path}'...")
    raw_df = pd.read_csv(input_data_path)

    # --- Step 2: Preprocess the raw data ---
    print("Preprocessing raw data to calculate probabilities...")
    processed_df = preprocess_pipeline_output(raw_df)
    print("Data processed successfully. DataFrame now has 'positive_probability' column.")
    print(processed_df.head())

    # --- Step 3: Test the plotting functions ---

    # Test 3a: ROC Curve
    print("\nTesting plot_roc_curve...")
    roc_output_path = os.path.join(output_dir, "dummy_roc_curve.png")
    plot_roc_curve(processed_df, output_path=roc_output_path)

    # Test 3b: Score Distribution
    # We need to split the data into groups to test this function properly
    print("\nTesting plot_score_distribution...")
    positive_df = processed_df[processed_df['labels'] == 1]
    negative_df = processed_df[processed_df['labels'] == 0]

    dist_data_dict = {
        'Positive Labels': positive_df,
        'Negative Labels': negative_df
    }
    dist_output_path = os.path.join(output_dir, "dummy_score_distribution.png")
    plot_score_distribution(dist_data_dict, output_path=dist_output_path)

    # Test 3c: Box Plot
    print("\nTesting plot_score_boxplot...")
    box_output_path = os.path.join(output_dir, "dummy_boxplot.png")
    plot_score_boxplot(processed_df, group_column='labels', output_path=box_output_path)

    print(f"\n--- All tests completed. Check the '{output_dir}/' directory for the output PNG files. ---")

if __name__ == "__main__":
    run_tests()
