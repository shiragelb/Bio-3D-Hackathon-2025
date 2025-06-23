# **NES-Finder: A Transformer-based Pipeline for Identifying Novel Nuclear Export Signals**

**A project for the "3D and 4D Structural Biology Data Processing" course (76562) Hackathon.**
* **Lecturers:** Prof. Dina Schneidman, Dr. Barak Raveh
* **Teaching Assistant:** Mr. Tomer Cohen

* **Team Members:** Daniel Levin, Imri Shuval, Shira Gelbstein, Ron Levin

### **Quick Links & Navigation**

* **Final Report:** For a comprehensive overview of the project's background, results, and conclusions, please see our report in `analysis/Bio Hackathon Report.pdf`.
* **Results & Analysis:** To see the code that generated all our figures and tables from the final data, please review the `analysis/analysis.ipynb` Jupyter Notebook. This is the reproducible proof of our findings.
* **Model Architecture:** The core deep learning model is defined in `src/nets/transformer_NES_classifier.py`.

### **Project Overview**

This project addresses the challenge of discovering novel **Nuclear Export Signals (NES)** within the human proteome. NES motifs are short amino acid sequences that act as "zip codes," directing proteins for export from the cell nucleus. While crucial for cellular function and often implicated in disease, these signals have a degenerate pattern, making them difficult to identify computationally.

To solve this, we developed **NES-Finder**, a deep learning pipeline that uses a Transformer-based classifier to predict whether a given peptide sequence is a functional NES. By training on known examples, our model learns the complex patterns of NES motifs and can screen thousands of proteins to find new, high-confidence candidates.

---

### **A Guided Tour for Reviewers**

We've designed this repository to be easy to navigate depending on your interests.

* ** If you want a high-level summary of our project and findings...
  * Start with the `analysis/Bio Hackathon Report.pdf`.
* ** If you want to see the final results and understand how we generated our plots...
  * The best place to go is the `analysis/analysis.ipynb` notebook.
* ** If you are interested in the deep learning model itself...
  * The model architecture is defined in the TransformerClassifier class inside `src/nets/transformer_NES_classifier.py`.
* ** If you want to understand the end-to-end data pipeline...
  * The entire workflow, from data ingestion to prediction, is orchestrated in `src/pipeline.py`.

### **Project Structure Guide**

Here is a breakdown of the key files and directories in this project:

* **`ex4_files/`**: Directory related to Exercise 4.
* **`data/`**: Contains all project data, including `DB_Tanya/` and `input_sequences/`.
* **`src/`**: Contains the core source code for the project.
    *   `pipeline.py`: The main data processing script.
    *   `train_model.py`: Handles the model training and evaluation logic.
    *   `nets/`: Contains the PyTorch implementations of the models.
    *   `plotting_utils.py`: A helper script containing standardized functions for creating visualizations.
    *   `scripts/`: Contains miscellaneous helper scripts for standalone tasks.
* **`outputs/`**: The default directory where all generated plots and data are saved.
* **`analysis/`**: Contains the final analysis and report.
    *   `analysis.ipynb`: **(Key file for results)** A Jupyter Notebook that serves as the reproducible record of our analysis.
    *   `Bio Hackathon Report.pdf`: **(Key file for overview)** The final, formal scientific report.
* **`README.md`**: This file.

---

### **Future Work & Ideas**

* **Structural Filtering:** Integrate 3D structural data from the AlphaFold DB to filter out candidate NES motifs that are not exposed on the protein's surface.  
* **Expanded Training Data:** Re-train the model on a larger dataset including different NES classes as they become available.  

---

### **How to Run the Project**

Follow these steps to set up the environment and run the analysis pipeline.

**1\. Setup Environment**

\# Clone the repository  
git clone github.com/shiragelb/Bio-3D-Hackathon-2025  
cd Bio-3D-Hackathon-2025

\# Create and activate a Python virtual environment (recommended)  
python \-m venv venv  
source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`

\# Install required dependencies  
pip install -r requirements.txt

**2\. Data Setup**

Place the necessary input data files inside the `data/input_sequences/` directory.

**3\. Run the Analysis**

The core analysis and visualizations can be reproduced without re-running the entire screening pipeline.

* Open the `analysis/analysis.ipynb` notebook in a Jupyter-compatible editor (like VS Code, PyCharm, or Jupyter Lab).
* Run the cells sequentially from top to bottom. This will load the pre-computed results, process them, and generate all plots in the `outputs/` directory.
