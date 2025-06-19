# **NES-Finder: A Transformer-based Pipeline for Identifying Novel Nuclear Export Signals**

**Hackathon 2025**

* **Team Members:** Daniel Levin, Imri Shuval, Shira Gelbstein, Ron Levin

### **Quick Links & Navigation**

* **Final Report:** For a comprehensive overview of the project's background, results, and conclusions, please see our [**scientific\_report.md**](http://docs.google.com/scientific_report.md).  
* **Results & Analysis:** To see the code that generated all our figures and tables from the final data, please review the [**analysis.ipynb**](http://docs.google.com/analysis.ipynb) Jupyter Notebook. This is the reproducible proof of our findings.  
* **Model Architecture:** The core deep learning model is defined in [**transformer\_NES\_classifier.py**](http://docs.google.com/transformer_NES_classifier.py).

### **Project Overview**

This project addresses the challenge of discovering novel **Nuclear Export Signals (NES)** within the human proteome. NES motifs are short amino acid sequences that act as "zip codes," directing proteins for export from the cell nucleus. While crucial for cellular function and often implicated in disease, these signals have a degenerate pattern, making them difficult to identify computationally.

To solve this, we developed **NES-Finder**, a deep learning pipeline that uses a Transformer-based classifier to predict whether a given peptide sequence is a functional NES. By training on known examples, our model learns the complex patterns of NES motifs and can screen thousands of proteins to find new, high-confidence candidates.

### **How to Run the Project**

Follow these steps to set up the environment and run the analysis pipeline.

**1\. Setup Environment**

\# Clone the repository  
git clone \[Your-Repository-URL\]  
cd \[repository-name\]

\# Create and activate a Python virtual environment (recommended)  
python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

\# Install required dependencies  
pip install \-r requirements.txt

**2\. Data Setup**

Place the necessary input data files inside the DB/ directory. For testing purposes, you can use the pipeline\_output\_dummy\_v2.csv file.

**3\. Run the Analysis**

The core analysis and visualizations can be reproduced without re-running the entire screening pipeline.

* Open the **analysis.ipynb** notebook in a Jupyter-compatible editor (like VS Code, PyCharm, or Jupyter Lab).  
* Run the cells sequentially from top to bottom. This will load the pre-computed results, process them, and generate all plots in the analysis\_outputs/ directory.

### **Project Structure Guide**

Here is a breakdown of the key files and directories in this project:

* **pipeline.py**: The main data processing script. It handles loading sequences, generating embeddings with ESM-2, running the classifier, and producing the final raw output file.  
* **transformer\_NES\_classifier.py**: Contains the PyTorch implementation of our TransformerClassifier model, which is the core of our prediction engine.  
* **plotting\_utils.py**: A helper script containing standardized functions for creating the project's main visualizations (ROC curves, box plots, etc.).  
* **analysis.ipynb**: **(Key file for results)** A Jupyter Notebook that serves as the reproducible record of our analysis. It loads the pipeline's output and uses plotting\_utils.py to generate all figures and tables shown in our report.  
* **scientific\_report.md**: **(Key file for overview)** The final, formal scientific report detailing the project's background, methods, results, and discussion.  
* **requirements.txt**: A list of all Python packages required to run the project.  
* **DB/**: A directory intended for storing input data files (e.g., FASTA files, NesDB datasets).  
* **analysis\_outputs/**: The default directory where all generated plots are saved.

### **A Guided Tour for Reviewers**

We've designed this repository to be easy to navigate depending on your interests.

* **If you want a high-level summary of our project and findings...**Start with the [**scientific\_report.md**](http://docs.google.com/scientific_report.md).  
* **If you want to see the final results and understand how we generated our plots...**The best place to go is the [**analysis.ipynb**](http://docs.google.com/analysis.ipynb) notebook. This is the executable proof of our work.  
* **If you are interested in the deep learning model itself...**The model architecture is defined in the TransformerClassifier class inside [**transformer\_NES\_classifier.py**](http://docs.google.com/transformer_NES_classifier.py).  
* **If you want to understand the end-to-end data pipeline...**The entire workflow, from data ingestion to prediction, is orchestrated in [**pipeline.py**](http://docs.google.com/pipeline.py).

### **Future Work & Ideas**

* **Structural Filtering:** Integrate 3D structural data from the AlphaFold DB to filter out candidate NES motifs that are not exposed on the protein's surface.  
* **Expanded Training Data:** Re-train the model on a larger dataset including different NES classes as they become available.  
* **Web Application:** Develop a simple web interface where a user can input a protein sequence and get back a list of predicted NES motifs.