# **Lexical semantic change detection a supervised approach**

This repository contains the work for my thesis on **Lexical Semantic Change Detection**, focusing on identifying and analyzing shifts in word meanings across time using modern NLP techniques. The thesis explores approaches like **Skip-Gram with Negative Sampling (SGNS)** on diachronic corpora.

## **Table of Contents**
- [Overview](#overview)
- [Datasets](#datasets)
- [Methods](#methods)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

---

## **Overview**
The goal of this thesis is to:
1. Detect lexical semantic changes using supervised approaches.
2. Analyze the semantic shifts of words across historical (1800s) and modern (2000s) corpora.
3. Develop methods to ensure high-quality annotations and robust analysis.

Key features:
- Use of **SemEval datasets** for benchmarking.
- Experiments with word embeddings and contextualized models.
- Annotated dataset creation for supervised classification.

---

## **Datasets**
### Sources:
1. **Cleaned Corpus of Historical American English (CCOHA)**:
   - **C1 (Historical Corpus)**: Covers 1810–1960.
   - **C2 (Modern Corpus)**: Covers 1960–present.

### Preprocessing:
- Part-of-speech (POS) tagging
- tokenization
- Stopword and punctuation removal
- Sentences formatted as lists with POS tags (e.g., `["word_nn", "run_vb"]`).

---

## **Methods**
1. **Unsupervised Candidate Word Generation**:
   - SGNS models: own-trained and pretrained
   - Embedding alignment: incremental learning, Orthogonal Procrustes, or both
   - Threshold strategies: average, median, lower confidence interval, or 5th percentile of cosine similarities for known semantic shifts
   - Semantic change detection via cosine similarity and alignment methods.

2. **Annotation Pipeline**:
   - Selection of representative sentences using clustering and contextual scoring.
   - Annotation validation through inter-annotator agreement analysis.
   - Conducted by three independent annotators
   - Classification criteria: Focused on moderate or strong semantic change. Majority vote used for reliability

3. **Supervised lexical semantic change detection**:
   - Developed a supervised model using the annotated dataset
   - Features: i) Word embeddings from each corpus ii) TF-IDF scores to capture semantic and contextual information


---

## **Repository Structure**
```plaintext
├── data/                  # Datasets (not included for licensing reasons)
├── notebooks/             # Jupyter Notebooks for experiments and EDA
├── src/                   # Source code for preprocessing, modeling, and evaluation
├── results/               # Results, visualizations, and outputs
├── docs/                  # Thesis documents and supporting materials
├── README.md              # Project overview and instructions
├── LICENSE                # License information
├── .gitignore             # Files to exclude from version control
