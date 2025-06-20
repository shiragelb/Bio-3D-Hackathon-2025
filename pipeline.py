import os
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from Ex4_files.clustering import kmeans_clustering, tsne_dim_reduction
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from Ex4_files.plot import plot_2dim_reduction
from create_data import create_data
from transformer_NES_classifier import get_transformer_classifier
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


def process_and_train(train_loader, val_loader, max_seq_len, emb_dim, device):
    model = get_transformer_classifier(
        max_seq_len=max_seq_len,
        esm_embedding_dim=emb_dim,
    ).to(device)

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


def save_predictions_to_csv(probabilities, predictions, labels, output_csv_path):
    probs_str = [str(p) for p in probabilities]
    output_df = pd.DataFrame({
        # "uniprotID": df["uniprotID"].tolist(),
        "logits": probs_str,
        "predictions": predictions,
        "labels": labels.tolist()
    })
    output_df.to_csv(output_csv_path, index=False)


def plot_embeds_in_2d(embeds, labels):
    train_np = np.array([emb.mean(0) for emb in embeds])
    k_means_labels = kmeans_clustering(train_np, k=2)
    coords_2d = tsne_dim_reduction(train_np, dim=2)

    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in labels], out_file_path="2d_true_labels.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path="2d_k_means.png")


def calc_test_prediction(predictions, labels):
    prediction_aggregated = torch.max(predictions, dim=1)[0]  # Get the max probability for each sample


def main_new():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_predictions_csv = "model_output.csv"

    # 1. Get embeddings and labels for training and test sets
    train_embeds, train_labels, test_embeds, test_labels = create_data()
    pos_count, neg_count = 0, 0
    for label in train_labels:
        if label == 0:
            neg_count += 1
        else:
            pos_count += 1
    print(f"Train set contains {pos_count} positive labels and {neg_count} negative labels.")

    # 2. Train model
    train_loader = data_to_loaders(train_embeds, train_labels, train_fraction=1)
    window_size = train_embeds.size(1)
    model = process_and_train(train_loader, val_loader=None, device=device,
                              max_seq_len=window_size, emb_dim=train_embeds.size(2))

    # 3. Run model on test set
    print("Running model on test set...")
    pos_count, neg_count = 0, 0
    for label in test_labels:
        if label == 0:
            neg_count += 1
        else:
            pos_count += 1
    print(f"Test set contains {pos_count} positive labels and {neg_count} negative labels.")
    test_pred_probs = []
    test_embeds, test_labels = test_embeds, test_labels
    for embed in tqdm(test_embeds):
        embed = embed.unsqueeze(0)  # Add batch dim
        logits, probs = model.predict_on_test(embed, W=window_size)
        logits, probs = torch.tensor(logits), torch.tensor(probs)
        prob_aggregated = torch.max(probs)  # Get the max probability for each sample
        test_pred_probs.append(prob_aggregated)

    threshold = 0.5
    predictions = np.array([1 if prob >= threshold else 0 for prob in test_pred_probs])
    save_predictions_to_csv(test_pred_probs, predictions, test_labels, output_predictions_csv)

    print("Training and testing complete. Predictions saved to", output_predictions_csv)
    test_labels = np.array(test_labels)
    acc = np.mean(predictions == test_labels)
    print("Test acc: ", acc)
    # TODO - calculate acc for only positives (label == 1)
    pos_mask = test_labels == 1  # positives
    pos_acc = (predictions[pos_mask] == 1).mean() if pos_mask.any() else float("nan")
    print("Fraction of samples predicted as positive: ", (predictions == 1).sum() / len(test_labels))
    print("Acc of positive samples: ", pos_acc)

    # TODO - calculate acc for only negative (label == 0)
    neg_mask = test_labels == 0  # negatives
    neg_acc = (predictions[neg_mask] == 0).mean() if neg_mask.any() else float("nan")
    print("Fraction of samples predicted as negative: ", (predictions == 0).sum() / len(test_labels))
    print("Acc of negative samples: ", neg_acc)


def main():
    pos_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    neg_csv_path = "input_sequences/PDB_Bacteria__Helical_Peptides_NESnegative_sequences.csv"
    combined_csv_path = "input_sequences/combined_dataset.csv"
    test_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    output_predictions_csv = "model_output.csv"

    # 1. Create combined dataset CSV with desired positive ratio
    create_dataset_from_csvs(pos_csv_path, neg_csv_path, output_csv_path=combined_csv_path)

    # 2. Extract embeddings and labels for training set
    train_embeddings, train_labels, train_df, device = extract_embeddings(combined_csv_path,
                                                                          embedding_path="train_embeddings.pt")
    plot_embeds_in_2d(train_embeddings, train_labels)

    # 3. Train the model on the train embeddings
    padded_embeddings = pad_sequence(train_embeddings, batch_first=True)
    train_loader, val_loader = data_to_loaders(padded_embeddings, train_labels)

    model = process_and_train(train_loader, val_loader, device=device,
                              max_seq_len=padded_embeddings.size(1), emb_dim=padded_embeddings.size(2))

    # 4. Extract embeddings for the test set
    test_embeddings, test_labels, test_df, _ = extract_embeddings(test_csv_path, embedding_path="test_embeddings.pt")

    # 5. Run the trained model on test embeddings
    W = -1  # Window size we trained on
    probs = model.predict_on_test(test_embeddings, W=W)
    threshold = 0.5
    predictions = [1 if prob >= threshold else 0 for prob in probs]
    save_predictions_to_csv(probs, predictions, test_df, output_predictions_csv)

    print("Training and testing complete. Predictions saved to", output_predictions_csv)


if __name__ == "__main__":
    main_new()
