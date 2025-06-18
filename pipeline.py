import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from transformer_NES_classifier import TransformerClassifier
from train_model import data_to_loaders, train

import random


def create_dataset_from_csvs(pos_csv, neg_csv, pos_ratio, output_csv_path, seed=42):
    pos_df = pd.read_csv(pos_csv)
    neg_df = pd.read_csv(neg_csv)


    # Calculate number of positive samples
    total_samples = len(pos_df) + len(neg_df)
    num_pos = int(total_samples * pos_ratio)
    num_neg = total_samples - num_pos

    # Sample positive and negative with given seed
    pos_sampled = pos_df.sample(n=num_pos, random_state=seed, replace=(num_pos > len(pos_df))).reset_index(drop=True)
    neg_sampled = neg_df.sample(n=num_neg, random_state=seed, replace=(num_neg > len(neg_df))).reset_index(drop=True)

    combined = pd.concat([pos_sampled, neg_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)
    combined.to_csv(output_csv_path, index=False)
    return output_csv_path


def extract_embeddings(csv_path, embedding_size=320, embedding_layer=6):
    df = pd.read_csv(csv_path)
    pep_tuples = list(zip(df["name"], df["full sequence"]))

    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    all_embeddings = get_esm_embeddings(
        pep_tuples,
        esm_model,
        alphabet,
        batch_converter,
        device,
        embedding_layer=embedding_layer,
        sequence_embedding=False
    )

    nes_embeddings = []
    for i, emb in enumerate(all_embeddings):
        start = df.iloc[i]["start#"]
        length = len(df.iloc[i].get("NES sequence", df.iloc[i].get("NOT NES", "")))  # support either column
        nes_emb = emb[start:start + length]
        nes_embeddings.append(torch.tensor(nes_emb, dtype=torch.float32))

    labels = torch.tensor(df["positive"].values, dtype=torch.long)
    return nes_embeddings, labels, df, device


def process_and_train(nes_embeddings, labels, device):
    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)

    model = TransformerClassifier(
        max_len=padded_embeddings.size(1),
        embedding_dim=320,
        positional_encoding="periodic_modulo",
        periods=(2, 3, 4),
        num_classes=2,
        pooling="cls",
        add_cls_token=True
    ).to(device)

    train_loader, val_loader = data_to_loaders(padded_embeddings, labels)
    train(model, train_loader, val_loader, device)
    return model


def predict(model, nes_embeddings):
    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        logits = model(padded_embeddings.to(device))
        predictions = torch.argmax(logits, dim=1)
    return logits, predictions


def save_predictions_to_csv(logits, predictions, df, output_csv_path):
    logits_str = [str(list(logit.detach().cpu().numpy())) for logit in logits]
    output_df = pd.DataFrame({
        "uniprotID": df["uniprotID"].tolist(),
        "logits": logits_str,
        "predictions": predictions.tolist(),
        "labels": df["label"].tolist()
    })
    output_df.to_csv(output_csv_path, index=False)

def main():
    pos_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    neg_csv_path = "input_sequences/PDB_Bacteria__Helical_Peptides_NESnegative_sequences.csv"
    combined_csv_path = "input_sequences/combined_dataset.csv"
    test_csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    output_predictions_csv = "model_output.csv"

    # 1. Create combined dataset CSV with desired positive ratio
    create_dataset_from_csvs(pos_csv_path, neg_csv_path, pos_ratio=0.5, output_csv_path=combined_csv_path)

    # 2. Extract embeddings and labels for training set
    train_embeddings, train_labels, train_df, device = extract_embeddings(combined_csv_path)

    # 3. Train the model on the training embeddings
    model = process_and_train(train_embeddings, train_labels, device)

    # 4. Extract embeddings for the test set
    test_embeddings, test_labels, test_df, _ = extract_embeddings(test_csv_path.head(10))

    # 5. Run the trained model on test embeddings 
    logits, predictions = predict(model, test_embeddings)

    # 6. saves the resultes in a csv file
    save_predictions_to_csv(logits, predictions, test_df, output_predictions_csv)   

    print("Training and testing complete. Predictions saved to", output_predictions_csv)


if __name__ == "__main__":
    main()
