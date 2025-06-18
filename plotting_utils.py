import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import ast


def preprocess_pipeline_output(df):
    """
    Takes the raw DataFrame from the pipeline and processes it for plotting.
    - Parses the string-formatted 'logits' column.
    - Converts logits to probabilities using softmax.
    - Adds a 'positive_probability' column.

    Args:
        df (pd.DataFrame): The DataFrame with columns ['uniprotID', 'logits', 'predictions', 'labels'].

    Returns:
        pd.DataFrame: The processed DataFrame with an added 'positive_probability' column.
    """
    # Safely parse the 'logits' column from string to list of floats
    # ast.literal_eval is safer than eval()
    logits_list = df['logits'].apply(ast.literal_eval).tolist()
    logits_array = np.array(logits_list)

    # Apply softmax to convert logits to probabilities
    # exp_logits = np.exp(logits_array)
    # probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    # A more numerically stable way to do softmax:
    exp_logits = np.exp(
        logits_array - np.max(logits_array, axis=1, keepdims=True))
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    # Create a new DataFrame with the results
    processed_df = df.copy()
    # Assume class 1 is the "positive" class
    processed_df['positive_probability'] = probabilities[:, 1]

    return processed_df


def plot_score_distribution(data_dict, score_column='positive_probability',
                            output_path="score_distribution.png"):
    """
    Plots histograms of the score distributions from one or more processed DataFrames.

    Args:
        data_dict (dict): A dictionary where keys are group names (e.g., "Positive Labels")
                          and values are the processed pandas DataFrames.
        score_column (str): The name of the column containing the scores to plot.
        output_path (str): Path to save the output PNG file.
    """
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    for group_name, df in data_dict.items():
        sns.histplot(df[score_column], bins=50, kde=True,
                     label=f'{group_name} (n={len(df)})', stat="density",
                     common_norm=False)

    plt.title('Distribution of Predicted NES Probabilities', fontsize=16)
    plt.xlabel('Predicted Probability of being an NES', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved score distribution plot to {output_path}")


def plot_roc_curve(df, score_column='positive_probability',
                   label_column='labels', output_path="roc_curve.png"):
    """
    Generates and plots a ROC curve from a processed DataFrame.

    Args:
        df (pd.DataFrame): Processed DataFrame containing scores and true labels.
        score_column (str): The name of the column containing the scores.
        label_column (str): The name of the column containing the true labels (0 or 1).
        output_path (str): Path to save the output PNG file.
    """
    plt.figure(figsize=(8, 8))

    y_true = df[label_column]
    y_scores = df[score_column]

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved ROC curve to {output_path}")


def plot_score_boxplot(df, group_column='labels',
                       score_column='positive_probability',
                       output_path="boxplot.png"):
    """
    Generates a box plot comparing score distributions, grouped by a column.

    Args:
        df (pd.DataFrame): Processed DataFrame containing scores and grouping info.
        group_column (str): Column to group by (e.g., 'labels' or 'source').
        score_column (str): Column containing the scores.
        output_path (str): Path to save the output PNG file.
    """
    plt.figure(figsize=(8, 8))
    sns.boxplot(x=group_column, y=score_column, data=df)

    plt.title('Comparison of NES Probability Scores', fontsize=16)
    plt.ylabel('Predicted Probability of being an NES', fontsize=12)
    plt.xlabel('Group', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved box plot to {output_path}")
