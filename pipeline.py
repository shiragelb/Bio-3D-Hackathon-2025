import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
<<<<<<< Updated upstream
from Ex4_files.esm_embeddings import get_esm_model, get_esm_embeddings
from transformer import TransformerClassifier  # Make sure your class is saved in transformer.py
=======
from esm_embeddings import get_esm_model, get_esm_embeddings
from transformer_NES_classifier import TransformerClassifier  
>>>>>>> Stashed changes

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

    return nes_embeddings, df

if __name__ == "__main__":
    csv_path = "NESdb_NESpositive_sequences.csv"  # replace with actual path

    # Step 1: Extract NES embeddings
    nes_embeddings, df = extract_nes_embeddings_from_csv(csv_path, embedding_size=320, embedding_layer=6)

    # Step 2: Pad to max length
    padded_embeddings = pad_sequence(nes_embeddings, batch_first=True)  # [batch_size, max_seq_len, embedding_dim]
    
    # Create dummy labels for testing
    labels = torch.zeros(len(nes_embeddings), dtype=torch.long)  # All 0s for now

    # Step 3: Initialize model
    model = TransformerClassifier(
        max_len=padded_embeddings.size(1),
        embedding_dim=320,
        positional_encoding="periodic_modulo",
        periods=(2, 3, 4),
        num_classes=2,
        pooling="cls",
        add_cls_token=True
    )

    # Step 4: Run model on the data
    with torch.no_grad():
        logits = model(padded_embeddings)
        predictions = torch.argmax(logits, dim=1)

    print("Logits:", logits)
    print("Predictions:", predictions)