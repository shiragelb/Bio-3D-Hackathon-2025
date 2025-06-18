import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from transformer_NES_classifier import TransformerClassifier
from train_model import data_to_loaders, train


def extract_nes_embeddings_from_csv(csv_path, embedding_size=320, embedding_layer=6):
    """
    Reads a CSV with full sequences and NES annotations, extracts NES embeddings using ESM.
    Returns a list of tensors, each of shape [nes_len, embedding_size]
    """
    df = pd.read_csv(csv_path).head(10)

    # Build input list for embedder
    pep_tuples = list(zip(df["name"], df["full sequence"]))

    # Load ESM model
    esm_model, alphabet, batch_converter, device = get_esm_model(embedding_size=embedding_size)

    # Get per-residue embeddings
    all_embeddings = get_esm_embeddings(
        pep_tuples,
        esm_model,
        alphabet,
        batch_converter,
        device,
        embedding_layer=embedding_layer,
        sequence_embedding=False
    )

    # Extract NES embeddings
    nes_embeddings = []
    for i, emb in enumerate(all_embeddings):
        start = df.iloc[i]["start#"]
        nes_len = len(df.iloc[i]["NES sequence"])
        nes_emb = emb[start:start + nes_len]  # shape: [nes_len, embedding_size]
        nes_embeddings.append(torch.tensor(nes_emb, dtype=torch.float32))

    labels = torch.tensor(df["positive"].values, dtype=torch.long)

    return nes_embeddings, df, labels, device

def main():
    csv_path = "input_sequences/NESdb_NESpositive_sequences.csv"
    # Step 1: Extract NES embeddings
    nes_embeddings, df, labels, device = extract_nes_embeddings_from_csv(csv_path, embedding_size=320,
                                                                         embedding_layer=6)

    # Step 2: Pad to max length
    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)  # [batch_size, max_seq_len, embedding_dim]

    # Step 3: Initialize model
    model = TransformerClassifier(
        max_len=padded_embeddings.size(1),
        embedding_dim=320,
        positional_encoding="periodic_modulo",
        periods=(2, 3, 4),
        num_classes=2,
        pooling="cls",
        add_cls_token=True
    ).to(device)

    # Step 4: Create loaders:
    train_loader, val_loader = data_to_loaders(padded_embeddings, labels)
    train(model, train_loader, val_loader, device)  # No optimizer needed for inference

if __name__ == '__main__':
    main()