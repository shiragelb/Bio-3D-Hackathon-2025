import json

import numpy as np
from neural_net import get_net_scores, train_net, prepare_loader, SimpleDenseNet
from pep_utils import load_peptide_data, get_peptide_distances, pep_train_test_split
from plot import plot_boxplot, plot_roc_curve, plot_esm_amino_acids_embeddings, plot_2dim_reduction
from esm_embeddings import get_esm_embeddings, get_esm_model
from clustering import kmeans_clustering, tsne_dim_reduction


def simple_score(p_train, n_train, p_test, n_test):
    """
    Distance-based baseline.
    For every peptide in test we compute its mean Euclidean distance
    to all positive-training embeddings and all negative-training
    embeddings.  The final score is a signed log-fold-difference:

        score  =  log1p(dist_to_neg) – log1p(dist_to_pos)

    So a higher score -> more “positive-like”, because
    distance-to-negatives is large while distance-to-positives is small.

    :param p_train: Positive-class train embeddings.
    :param n_train: Negative-class train embeddings.
    :param p_test: Positive-class test embeddings.
    :param n_test: Negative-class test embeddings.
    :return: Scores for the positive-test set, Scores for the negative-test set (np.ndarray, np.ndarray).
    """
    positive_mean_distances_pos = get_peptide_distances(p_test, p_train, reduce_func=np.mean)
    positive_mean_distances_neg = get_peptide_distances(p_test, n_train, reduce_func=np.mean)
    negative_mean_distances_pos = get_peptide_distances(n_test, p_train, reduce_func=np.mean)
    negative_mean_distances_neg = get_peptide_distances(n_test, n_train, reduce_func=np.mean)

    p_score = np.log1p(positive_mean_distances_neg) - np.log1p(positive_mean_distances_pos)
    # n_score = None  # TODO: fill this line
    n_score = np.log1p(negative_mean_distances_neg) - np.log1p(negative_mean_distances_pos)
    return p_score, n_score


def main(chosen_embedding_size=1280, batch_size=64, epochs=50, lr=1e-3, hidden_dim=128, dropout=0.2):
    # TODO: play with these parameters
    # ESM embedding dim (320-5120), Options are: 320, 480, 640, 1280, 2560, 5120.
    # which transformer layer to take for the embedding
    chosen_test_size = 0.25  # train/test split
    ESM_EMBED_SIZE_TO_LAYER = {320: 6,
                               480: 12,
                               640: 30,
                               1280: 33,
                               2560: 36,
                               5120: 48}
    chosen_embedding_layer = ESM_EMBED_SIZE_TO_LAYER[chosen_embedding_size]

    # Load all the peptide data. Note: The data, peptides, is represented by AA sequences
    print("Loading peptide data")
    positive_pep, negative_pep, doubt_labels = load_peptide_data()
    # Notes - positives are peptides with a Nuclear Export Signal (NES)
    #       - doubt_labels are labels we're not sure about (test set)

    # Load the pre-trained ESM-2 model with the desired embedding size
    print("Loading ESM-2 model")
    model_esm, alphabet_esm, batch_converter_esm, device_esm = get_esm_model(embedding_size=chosen_embedding_size)

    print("Getting ESM-2 amino acid embeddings for the first negative and positive peptides")
    # Get amino acids embedding of the first negative and first positive peptides
    positive_aa_embeddings = \
        get_esm_embeddings(positive_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                           embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]
    negative_aa_embeddings = \
        get_esm_embeddings(negative_pep[0:1], model_esm, alphabet_esm, batch_converter_esm, device_esm,
                           embedding_layer=chosen_embedding_layer, sequence_embedding=False)[0]
    print("Plotting heatmaps of the amino acid embedding")
    plot_esm_amino_acids_embeddings(positive_aa_embeddings, out_file_path="positive_heatmap.png")
    plot_esm_amino_acids_embeddings(negative_aa_embeddings, out_file_path="negative_heatmap.png")

    # Get the ESM-2 sequence embeddings from all the negative and positive peptides
    print("Getting the ESM-2 embeddings for all the peptide data")
    positive_esm_emb = get_esm_embeddings(positive_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)
    negative_esm_emb = get_esm_embeddings(negative_pep, model_esm, alphabet_esm, batch_converter_esm, device_esm,
                                          embedding_layer=chosen_embedding_layer, sequence_embedding=True)

    # Clustering and Dimensionality reduction
    print("Clustering using K-means clustering and reduce to dim=2 using TSNE")
    all_esm_embeddings = negative_esm_emb + positive_esm_emb
    all_labels = [0] * len(negative_esm_emb) + [1] * len(positive_esm_emb)

    k_means_labels = kmeans_clustering(all_esm_embeddings, k=2)
    coords_2d = tsne_dim_reduction(all_esm_embeddings, dim=2)

    print("Plotting 2D dimensionality reduction by true labels and by K-means clustering")
    plot_2dim_reduction(coords_2d, [["N", "P"][i] for i in all_labels], out_file_path="2d_true_labels.png")
    plot_2dim_reduction(coords_2d, k_means_labels, out_file_path="2d_k_means.png")

    # Split the data into train and test sets
    print("Splitting to train and test sets")
    positive_train, negative_train, is_doubt_train, positive_test, negative_test, is_doubt_test = pep_train_test_split(
        positive_esm_emb, negative_esm_emb, doubt_labels, test_size=chosen_test_size)

    print("Calculating euclidean distances of ESM-2 embeddings")
    # Calculate a score based on the log fold difference between distance to negative mean and positive mean
    positive_score, negative_score = simple_score(positive_train, negative_train, positive_test, negative_test)

    # Plot the results in a boxplot and in a ROC curve
    print("Plotting Baseline results")
    plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score},
                 out_file_path=f"baseline_boxplot_emb-size{chosen_embedding_size}.png")
    plot_roc_curve([0] * len(negative_score) + [1] * len(positive_score),
                   np.concatenate([negative_score, positive_score]),
                   out_file_path=f"baseline_roc_curve_emb-size{chosen_embedding_size}.png")

    # Train a simple neural network
    print("Training a neural network on the training set")
    # TODO: Select parameters for the network
    # Prepare a Dataloader and create model
    net_dataloader = prepare_loader(positive_train, negative_train, batch_size=batch_size)
    network = SimpleDenseNet(esm_emb_dim=chosen_embedding_size, hidden_dim=hidden_dim, dropout=dropout)
    trained_network = train_net(network, net_dataloader, num_epochs=epochs, lr=lr)

    print("Getting the trained network scores for the test set")
    positive_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=positive_test)
    negative_score = get_net_scores(trained_net=trained_network, esm_seq_embeddings=negative_test)

    print("Plotting Network results")
    plot_boxplot({"Positive Test": positive_score, "Negative Test": negative_score},
                 out_file_path=f"network_boxplot_emb{chosen_embedding_size}_bs{batch_size}_e{epochs}_lr{lr}_hdim{hidden_dim}_do{dropout}.png")
    roc = plot_roc_curve([0] * len(negative_score) + [1] * len(positive_score),
                         np.concatenate([negative_score, positive_score]),
                         out_file_path=f"network_roc_curve_emb{chosen_embedding_size}_bs{batch_size}_e{epochs}_lr{lr}_hdim{hidden_dim}_do{dropout}.png")

    return {"AUC": roc, "chosen_embedding_size": chosen_embedding_size, "batch_size": batch_size, "epochs": epochs,
            "lr": lr, "hidden_dim": hidden_dim, "dropout": dropout}


def grid_search():
    from itertools import product

    batch_size_ = [32, 64, 128]
    epochs_ = [50, 100]
    lr_ = [1e-3]
    hidden_dim_ = [128, 256, 512]
    dropout_ = [0.2, 0.3, 0.4, 0.5]
    chosen_embedding_size_ = [320, 480, 640, 1280, 2560]
    param_names = ["batch_size", "epochs", "lr", "hidden_dim", "dropout", "chosen_embedding_size"]
    param_ranges = [
        batch_size_,
        epochs_,
        lr_,
        hidden_dim_,
        dropout_,
        chosen_embedding_size_,
    ]
    params = [dict(zip(param_names, values)) for values in product(*param_ranges)]
    results: list[dict] = []
    for param_dict in params:
        results.append(main(**param_dict))
        # Update json:
        with open("results_log.json", "w") as f:
            json.dump(results, f, indent=2)

    max_AUC: dict = max(results, key=lambda x: x.get("AUC", float('-inf')))
    print("Highest AUC achieved is: ", max_AUC.pop("AUC"))
    print("Corresponding params: ", max_AUC)


if __name__ == '__main__':
    best_params = {  # "AUC": 0.924931129476584,
        "chosen_embedding_size": 2560,
        "batch_size": 32,
        "epochs": 50,
        "lr": 0.001,
        "hidden_dim": 512,
        "dropout": 0.4
    }
    main(**best_params)
