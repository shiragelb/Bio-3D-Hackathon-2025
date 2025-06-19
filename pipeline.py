from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from Ex4_files.clustering import kmeans_clustering, tsne_dim_reduction
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from Ex4_files.plot import plot_2dim_reduction
from transformer_NES_classifier import TransformerClassifier, get_transformer_classifier
from train_model import data_to_loaders, train, evaluate

import random


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


def extract_embeddings(csv_path, embedding_size=320, embedding_layer=6, embedding_path: str = "embeddings.pt"):
    if Path(embedding_path).exists():
        print("Loading embeddings from", embedding_path)
        embeddings_data = torch.load(embedding_path, weights_only=False)
        return embeddings_data['nes_embeddings'], embeddings_data['labels'], embeddings_data['df'], embeddings_data[
            'device']

    print("Extracting embeddings from", csv_path)
    df = pd.read_csv(csv_path)
    df.drop(columns=['name', 'Unnamed: 6'], inplace=True)
    pep_tuples = list(zip(df["uniprotID"], df["full sequence"]))

    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    nes_embeddings = []
    embed_batch_size = 8  # Adjust batch size, according to your GPU memory. 6GB VRam appears to handle 8.
    total = len(pep_tuples)
    current = 0
    bad_labels = []
    while current < total:
        if current + embed_batch_size > total:  # Last batch may be smaller
            embed_batch_size = total - current
        print(f"Processing embedding {current + embed_batch_size}/{total}...")
        batch = pep_tuples[current:current + embed_batch_size]

        # Get embeddings for the current batch
        batch_embeddings = get_esm_embeddings(
            batch,
            esm_model,
            alphabet,
            batch_converter,
            device,
            embedding_layer=embedding_layer,
            sequence_embedding=False
        )

        for i, emb in enumerate(batch_embeddings):
            i = i + current
            start = df.iloc[i]["start#"]
            try:
                if df.iloc[i]["label"] == 1:
                    length = len(df.iloc[i]["NES sequence"])
                else:
                    length = len(df.iloc[i]["NOT NES"])
                nes_emb = emb[start:start + length]
                nes_embeddings.append(torch.tensor(nes_emb, dtype=torch.float32))
            except TypeError:
                print("error with sequence at index", i)
                bad_labels.append(i)

        current += embed_batch_size

    df.drop(index=bad_labels, inplace=True)  # drop rows with bad labels
    # convert labels to tensor of shape (N, 2):
    labels = torch.tensor([[0, 1] if label == 1 else [1, 0] for label in df["label"].values], dtype=torch.float32)
    torch.save({
        'nes_embeddings': nes_embeddings,
        'labels': labels,
        'df': df,
        'device': device
    }, embedding_path)

    return nes_embeddings, labels, df, device


def process_and_train(train_loader, val_loader, max_seq_len, emb_dim, device):
    model = get_transformer_classifier(
        max_seq_len=max_seq_len,
        esm_embedding_dim=emb_dim,
    ).to(device)

    print("Evaluating model before training...")
    val_loss, val_acc, predictions, probabilities, labels = evaluate(model, val_loader, device)
    print("Validation Loss:", val_loss, "Validation Accuracy:", val_acc)
    train(model, train_loader, val_loader, device)
    print("\nEvaluating model post training...")
    val_loss, val_acc, predictions, probabilities, labels = evaluate(model, val_loader, device)
    print("Validation Loss:", val_loss, "Validation Accuracy:", val_acc)
    print("Avg probability of positive class:", probabilities[:, 1].mean().item())
    print("Avg probability of negative class:", probabilities[:, 0].mean().item())
    # Calculating false positive and false negative rates:
    false_positive_rate = (predictions[labels[:, 0] == 1] == 1).sum().item() / (labels[:, 0] == 1).sum().item()
    false_negative_rate = (predictions[labels[:, 1] == 1] == 0).sum().item() / (labels[:, 1] == 1).sum().item()
    print("False Positive Rate:", false_positive_rate)
    print("False Negative Rate:", false_negative_rate)
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


def save_predictions_to_csv(logits, predictions, df, output_csv_path):
    logits_str = [str(list(logit.detach().cpu().numpy())) for logit in logits]
    output_df = pd.DataFrame({
        "uniprotID": df["uniprotID"].tolist(),
        "logits": logits_str,
        "predictions": predictions.tolist(),
        "labels": df["label"].tolist()
    })
    output_df.to_csv(output_csv_path, index=False)


def plot_embeds_in_2d(embeds, labels):
    train_np = np.array([emb.mean(0) for emb in embeds])
    # labels are currently in [0/1, 0/1] format, convert to 0/1:
    labels = [1 if label[1] == 1 else 0 for label in labels]
    k_means_labels = kmeans_clustering(train_np, k=2)
    coords_2d = tsne_dim_reduction(train_np, dim=2)

    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in labels], out_file_path="2d_true_labels.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path="2d_k_means.png")


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
    logits, probabilities, predictions = predict(model, test_embeddings)

    # 6. saves the results in a csv file
    save_predictions_to_csv(logits, predictions, test_df, output_predictions_csv)

    print("Training and testing complete. Predictions saved to", output_predictions_csv)


if __name__ == "__main__":
    main()