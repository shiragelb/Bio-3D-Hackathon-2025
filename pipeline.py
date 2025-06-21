import os
import random
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from Ex4_files.clustering import kmeans_clustering, tsne_dim_reduction
from create_data import create_data
from nets.FF_classifier import get_FF_classifier
from nets.transformer_NES_classifier import get_transformer_classifier
from train_model import data_to_loaders, train, evaluate
from create_data import extract_embeddings


def create_dataset_from_csvs(pos_csv, neg_csv, output_csv_path):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)

    # Calculate number of positive samples
    # total_samples = len(pos_df) + len(neg_df)
    # num_pos = int(total_samples * pos_ratio)
    # num_neg = total_samples - num_pos

    # Sample positive and negative with given seed
    # pos_sampled = pos_df.sample(n=num_pos, random_state=seed, replace=(num_pos > len(pos_df))).reset_index(drop=True)
    # neg_sampled = neg_df.sample(n=num_neg, random_state=seed, replace=(num_neg > len(neg_df))).reset_index(drop=True)

    combined = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    combined.to_csv(output_csv_path, index=False)
    return output_csv_path


def get_model(model_type, max_seq_len, emb_dim, device):
    if model_type == "transformer_classifier":
        print("Using a transformer classifier")
        model = get_transformer_classifier(
            max_seq_len=max_seq_len,
            esm_embedding_dim=emb_dim,
        ).to(device)
    elif model_type == "FF_classifier":
        print("Using a simple feed-forward network")
        model = get_FF_classifier(seq_len=max_seq_len, emb_dim=emb_dim).to(device)
    else:
        raise ValueError("Unknown model type")
    return model


def process_and_train(model, train_loader, val_loader, device):
    if val_loader:
        print("Evaluating model before training...")
        val_loss, val_acc, preds, probabilities, labels = evaluate(model, val_loader, device)
        print("Validation Loss:", val_loss, "Validation Accuracy:", val_acc)
        print("Avg probability given :", probabilities.mean().item())

    os.makedirs("models", exist_ok=True)
    train(model, train_loader, val_loader, device, save_path="models/T_classifier.pt")
    print("\nEvaluating model post training...")
    if val_loader:
        val_loss, val_acc, preds, probabilities, labels = evaluate(model, val_loader, device)
        print("Validation Loss:", val_loss, "Validation Accuracy:", val_acc)
        print("Avg probability given :", probabilities.mean().item())
        # Calculating false positive and false negative rates:
        tp = np.sum((preds == 1) & (labels == 1))  # true positives count
        tn = np.sum((preds == 0) & (labels == 0))  # true negatives count
        fp = np.sum((preds == 1) & (labels == 0))  # false positives count
        fn = np.sum((preds == 0) & (labels == 1))  # false negatives count
        # Now calculate rates:
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"False Positive Rate (FPR): {fpr:.3%}")
        print(f"False Negative Rate (FNR): {fnr:.3%}")
    return model


def predict(model, nes_embeddings):
    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        logits = model(padded_embeddings.to(device))
        probabilities = torch.sigmoid(logits)
        predictions = torch.argmax(logits, dim=1)
    return logits, probabilities, predictions


def save_predictions_to_csv(scores, predictions, labels, test_ids, test_sequences, output_csv_path):
    """
    Scores are between 0 and 1.
    The predictions are based on a threshold of score>=0.5 == positive prediction, else negative.
    """
    scores_str = [str(s) for s in scores]
    output_df = pd.DataFrame({
        "score": scores_str,
        "predictions (threshold 0.5)": predictions,
        "labels": labels.tolist(),
        "uniprotID": test_ids,
        "sequence": test_sequences
    })
    output_df.to_csv(output_csv_path, index=False)


def plot_2dim_reduction(lower_dim_coords, labels, out_file_path):
    """
    Scatter plot 2D coordinates with points colored by `labels`.
    :param lower_dim_coords: 2D coordinates
    :param labels: Class or cluster labels used to color points.
    :param out_file_path: The path for the output png
    """

    unique_labels = np.unique(labels)
    labels = np.array(labels)
    lower_dim_coords = np.array(lower_dim_coords)

    plt.figure(figsize=(6, 5))
    for lab in unique_labels:
        idx = (labels == lab)
        plt.scatter(
            lower_dim_coords[idx, 0],
            lower_dim_coords[idx, 1],
            label=str(lab),
        )

    plt.xlabel("DIM-1")
    plt.ylabel("DIM-2")
    plt.legend(title="Label", loc="best")
    plt.tight_layout()

    plt.savefig(out_file_path)
    plt.close()


def plot_embeds_in_2d(embeds, labels, plot_id: str):
    labels = np.array(labels,dtype=int)
    train_np = np.array([emb.mean(0) for emb in embeds])
    print(f"Plotting 2D dimensionality reduction of {plot_id} by true labels and by K-means clustering")
    k_means_labels = kmeans_clustering(train_np, k=2)
    coords_2d = tsne_dim_reduction(train_np, dim=2)
    os.makedirs("plots", exist_ok=True)
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in labels],
                        out_file_path=f"plots/2d_true_labels_{plot_id}.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path=f"plots/2d_k_means_{plot_id}.png")

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trained_model(model_type, train_embeds, train_labels, window_size, embed_dim, device,
                      load_prepared_model, model_savepath):
    """
    If a trained model already exists, this function will return it.
    Otherwise, it trains a new model and returns it.
    """
    train_loader = data_to_loaders(train_embeds, train_labels, train_fraction=1)
    model = get_model(model_type=model_type, max_seq_len=window_size, emb_dim=embed_dim, device=device)
    if load_prepared_model:
        saved = torch.load(model_savepath)
        state_dict = saved["model_state_dict"]
        model.load_state_dict(state_dict)
    else:  # Train
        model = get_model(model_type=model_type, max_seq_len=window_size, emb_dim=embed_dim, device=device)
        model = process_and_train(model=model, train_loader=train_loader, val_loader=None, device=device)
        # Save model:
        os.makedirs("models", exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),  # weights
                "max_seq_len": window_size,  # hyper-params needed to rebuild
                "emb_dim": train_embeds.size(2),
            },
            model_savepath
        )
    return model


def main():
    set_seeds()

    model_type = ["FF_classifier", "transformer_classifier"][1]
    load_prepared_model = [True, False][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get embeddings and labels for training and test sets
    train_embeds, train_labels, test_embeds, test_labels, test_ids, test_sequences = create_data()
    plot_embeds_in_2d(train_embeds, train_labels, plot_id="train-embeds")
    plot_embeds_in_2d(test_embeds, test_labels, plot_id="test-embeds")
    window_size = train_embeds.size(1)
    embed_dim = train_embeds.size(2)
    pos_count, neg_count = 0, 0
    for label in train_labels:
        if label == 0:
            neg_count += 1
        else:
            pos_count += 1
    print(f"Train set contains {pos_count} positive labels and {neg_count} negative labels.")

    # 2. Get trained model or load a trained one
    model_identifier = f"{model_type}_w-{window_size}_emb-{embed_dim}"
    model_savepath = f"models/{model_identifier}.pt"
    model = get_trained_model(model_type, train_embeds, train_labels, window_size, embed_dim, device,
                              load_prepared_model, model_savepath)

    # 3. Run model on test set
    output_predictions_csv = f"test_output_{model_identifier}.csv"
    print("Running model on test set...")
    pos_count, neg_count = 0, 0
    for label in test_labels:
        if label == 0:
            neg_count += 1
        else:
            pos_count += 1
    print(f"Test set contains {pos_count} positive labels and {neg_count} negative labels.")
    test_pred_scores = []
    for embed in tqdm(test_embeds):
        embed = embed.unsqueeze(0)  # Add batch dim
        logits, score = model.predict_on_test(embed, W=window_size)
        logits, score = torch.tensor(logits), torch.tensor(score)
        scores_aggregated = torch.max(score)  # Get the max probability for each sample
        test_pred_scores.append(scores_aggregated)
    test_pred_scores = [t.item() for t in test_pred_scores]
    threshold = 0.5
    predictions = np.array([1 if prob >= threshold else 0 for prob in test_pred_scores])
    save_predictions_to_csv(test_pred_scores, predictions, test_labels, test_ids, test_sequences,
                            output_predictions_csv)

    print("Training and testing complete. Predictions saved to", output_predictions_csv)
    test_labels = np.array(test_labels)
    acc = np.mean(predictions == test_labels)
    print("Test acc: ", acc)
    pos_mask = test_labels == 1  # positives
    pos_acc = (predictions[pos_mask] == 1).mean() if pos_mask.any() else float("nan")
    print("Fraction of samples predicted as positive: ", (predictions == 1).sum() / len(test_labels))
    print("Acc of positive samples: ", pos_acc)

    neg_mask = test_labels == 0  # negatives
    neg_acc = (predictions[neg_mask] == 0).mean() if neg_mask.any() else float("nan")
    print("Fraction of samples predicted as negative: ", (predictions == 0).sum() / len(test_labels))
    print("Acc of negative samples: ", neg_acc)


if __name__ == "__main__":
    main()
