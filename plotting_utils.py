import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc


def plot_score_distribution(scores_dict, output_path="score_distribution.png"):
    """
    Plots histograms of the score distributions for different groups.
    This helps visualize how well the model separates different datasets.

    Args:
        scores_dict (dict): A dictionary where keys are group names (e.g., "Human Proteome")
                            and values are lists of the positive probability scores.
        output_path (str): Path to save the output PNG file.
    """
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    for group_name, scores in scores_dict.items():
        # Use kde=True to show a smoothed density curve over the histogram
        sns.histplot(scores, bins=50, kde=True,
                     label=f'{group_name} (n={len(scores)})', stat="density",
                     common_norm=False)

    plt.title('Distribution of Predicted NES Probabilities', fontsize=16)
    plt.xlabel('Predicted Probability of being an NES', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.xlim(0, 1)  # Probabilities are between 0 and 1
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved score distribution plot to {output_path}")


def plot_roc_curve(true_positive_scores, true_negative_scores,
                   output_path="roc_curve.png"):
    """
    Generates and plots a Receiver Operating Characteristic (ROC) curve.
    This function requires a set of known positives and known negatives to evaluate
    the classifier's performance.

    Args:
        true_positive_scores (list): A list of positive probability scores from a known positive set
                                     (e.g., the original NesDB positives).
        true_negative_scores (list): A list of positive probability scores from a known negative set
                                     (e.g., the mitochondrial proteins).
        output_path (str): Path to save the output PNG file.
    """
    plt.figure(figsize=(8, 8))

    # Combine scores and create true labels
    y_scores = true_positive_scores + true_negative_scores
    y_true = [1] * len(true_positive_scores) + [0] * len(true_negative_scores)

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


def plot_score_boxplot(scores_dict, output_path="boxplot.png"):
    """
    Generates a box plot to compare the statistical summary of score
    distributions from different groups.

    Args:
        scores_dict (dict): A dictionary where keys are group names
                            (e.g., "Human Proteome", "Negative Controls")
                            and values are lists of positive probability scores.
        output_path (str): Path to save the output PNG file.
    """
    # Convert the dictionary to a pandas DataFrame for easier plotting with seaborn.
    # This format is called "long-form" data.
    plot_data = []
    for group_name, scores in scores_dict.items():
        for score in scores:
            plot_data.append({'Group': group_name, 'Score': score})
    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 8))
    sns.boxplot(x='Group', y='Score', data=df)

    plt.title('Comparison of NES Probability Scores', fontsize=16)
    plt.ylabel('Predicted Probability of being an NES', fontsize=12)
    plt.xlabel('Data Source', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved box plot to {output_path}")
