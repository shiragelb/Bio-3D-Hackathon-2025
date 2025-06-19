# Screening the Human Proteome for Novel Nuclear Export Signals using a Transformer-based Deep Learning Model

**Authors:** Daniel Levin, Imri Shuval, Shira Gelbstein, Ron Levin

### Abstract

*(TODO: This section should be 150-250 words and written last, but it's good to have a template)*

**Background:** The transport of proteins between the cell nucleus and the cytoplasm is a key process for cellular function. This transport is mediated by specific signals, including the Nuclear Export Signal (NES), which directs proteins for export from the nucleus through the CRM1 pathway. While a consensus pattern for NES sequences is known, it is degenerate, making the computational identification of novel NES motifs a challenge.

**Problem:** A large number of proteins may contain functional NES motifs that have not yet been discovered. Computational tools can help to systematically scan entire proteomes to identify these candidates for further study.

**Approach:** We developed a computational pipeline to address this challenge. Our method represents protein sequences using embeddings from the ESM-2 protein language model. These embeddings are then used to train a Transformer-based classifier to distinguish between NES and non-NES peptides. We deployed this pipeline to screen the entire human proteome.

**Results:** Our classifier demonstrated good performance, achieving an Area Under the Curve (AUC) of **\[Enter AUC value, e.g., 0.91\]** on a held-out test set. The full proteome screen identified **\[Enter number, e.g., 158\]** high-confidence novel NES candidates (prediction score \> 0.9). The distribution of scores for our candidate pool was different from that of a negative control set, supporting the specificity of our method.

**Conclusion:** Our work presents a tool for the discovery of novel NES motifs. The high-confidence candidates we identified provide a resource for experimental biologists, which may help uncover new regulatory mechanisms.

### End Of Abstract

### 1\. Introduction

#### General Biological Background

Living organisms are made of cells, the basic units of life. In complex organisms like humans, these are called eukaryotic cells. A key feature of these cells is a compartment called the **nucleus**, which encloses the cell's genetic blueprint, DNA.

The site of most protein synthesis and cellular activities is the **cytoplasm**, the region outside the nucleus. For the cell to function, there must be a constant, regulated flow of molecules between the nucleus and the cytoplasm. This traffic moves through gateways in the nuclear membrane called **Nuclear Pore Complexes (NPCs)**.

#### Specific Background: The NES Signal

For a protein to exit the nucleus through an NPC, it often needs to carry a specific signal. This signal is a short amino acid sequence within the protein called a **Nuclear Export Signal (NES)**. The NES motif serves as a binding site for a transport protein called **CRM1** (also known as XPO1). When CRM1 binds to a protein's NES, it transports that protein out of the nucleus into the cytoplasm.

This process is important for normal cell function and is also implicated in diseases. For instance, many viruses use the CRM1 pathway to export their own proteins, and in some cancers, CRM1 is overactive. Because of this, CRM1 is a therapeutic target in several diseases.

A key challenge is that the NES is not a single, fixed sequence. It is a degenerate pattern, generally rich in hydrophobic (water-repelling) amino acids like Leucine (L) at specific spacings. This ambiguity makes it difficult to reliably search for NES motifs in a protein sequence.

#### Project Goal

The main objective of this project is to develop and apply a deep learning-based pipeline to address the challenge of ambiguous NES patterns. By learning from known examples, we aim to create an accurate classifier that can scan the entire human proteome (\~20,000 proteins) and identify novel, high-confidence NES candidates.

**Key Citations:**

* \[Xu, D., et al. (2012). MBoC.\]  
* \[Fung, H. Y. J., et al. (2021). MBoC.\]

### 2\. Methods

#### Dataset and Preprocessing

Our model was trained and validated using data from the NesDB database. We curated a set of **\[Number\]** positive NES sequences and **\[Number\]** negative sequences, sourced from mitochondrial proteins, to form our control group. For the final screening, we used the canonical human proteome dataset from UniProt (Release 2024\_03).

#### Computational Approach

Our pipeline consists of two main stages:

**1\. Sequence Representation:** We first convert protein sequences into a numerical format using the **ESM-2 protein language model**. This model generates per-residue embeddings that capture contextual information about each amino acid.

**2\. Classification Model:** The sequence embeddings are then used as input to our **Transformer-based classifier** (transformer\_NES\_classifier.py). A Transformer architecture is well-suited for this task as its self-attention mechanism can identify long-range dependencies and spacing patterns in the sequence. The model was trained using a binary cross-entropy loss function.

**3\. Large-Scale Screening:** We implemented a sliding window approach for the full proteome screen. A window of 22 amino acids was moved across each of the \~20,000 proteins, and each window was passed through our trained classifier to generate a prediction score.

***\[Placeholder for a simple flowchart diagram. You can create this in PowerPoint or a diagram tool and insert it as an image here. It should show: FASTA File \-\> Sliding Window \-\> ESM-2 Embedding \-\> Transformer Classifier \-\> Results CSV\]***

All code for this project is available on our GitHub repository: **\[Link to your GitHub repository\]**

### 3\. Results (Experiments)

#### Model Performance Validation

First, we assessed the classification performance of our model on a test set of known positive and negative controls. The model's ability to discriminate between the two classes is visualized in the Receiver Operating Characteristic (ROC) curve below.

***\[Placeholder for Figure 1: Your ROC Curve plot. The caption should read: "Figure 1: ROC curve for the Transformer classifier on the validation set. The model achieved an AUC of \[Your AUC Value\], demonstrating good discriminative ability."\]***

#### Human Proteome Screen Analysis

Next, we analyzed the results of the full proteome screen by comparing the distribution of prediction scores from the human proteome against the scores from our negative control set.

***\[Placeholder for Figure 2: Your Score Distribution plot. The caption should read: "Figure 2: Distribution of predicted NES probabilities. The human proteome screen (blue) shows a tail of high-scoring candidates compared to the negative control set (orange), which is concentrated near zero."\]***

***\[Placeholder for Figure 3: Your Box Plot. The caption should read: "Figure 3: Box plot comparing score distributions. The median score for the human proteome hits is higher than for the negative controls, supporting the specificity of our predictions."\]***

From the full screen, we identified **\[Number\]** high-confidence candidates by applying a probability threshold of 0.90. The top 10 candidates are listed in Table 1\.

***\[Placeholder for Table 1: A clean table showing your top 10 hits, with columns for Protein ID, Start Position, Sequence, and Score.\]***

### 4\. Discussion

Our project developed a deep learning pipeline for the large-scale discovery of Nuclear Export Signals. Our model demonstrated good accuracy (AUC \= **\[Your AUC Value\]**), and its application to the human proteome yielded **\[Number\]** new candidates.

Strengths and Insights:  
A strength of our approach is the use of a Transformer architecture on protein language model embeddings. This allows the model to learn context-dependent features of a functional NES. Our top candidates (Table 1\) provide a list for potential experimental validation.  
Limitations:  
Our model is sequence-based and does not consider the 3D structure of the protein. A functional NES must be accessible on the protein's surface to bind CRM1. It is possible that some of our high-scoring candidates are buried within the protein's core and are therefore non-functional. Additionally, our training data was limited to known NES motifs, which may not represent the full diversity of all NES classes.  
Future Work:  
A clear next step is to integrate structural information. After identifying a high-scoring sequence, one could fetch its predicted 3D structure from the AlphaFold Database and calculate the surface accessibility of the candidate peptide. This could filter out buried sequences and improve the quality of the candidate list.  
Broader Implications:  
The ability to map functional motifs across proteomes has implications for understanding protein transport. Our tool can aid research into these regulatory networks. By identifying new NES-containing proteins, we can identify new proteins involved in cellular pathways and provide potential new targets for therapies aimed at modulating the CRM1 export pathway.