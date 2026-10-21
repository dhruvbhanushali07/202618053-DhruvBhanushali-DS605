# DS605 Lab 7: Image Representation, PCA, Eigenfaces, and Face Recognition

---

## 🚀 Project Overview

This project explores image representation and dimensionality reduction using Principal Component Analysis (PCA) on the **ORL (Olivetti Research Laboratory) / AT&T Face Database**.

The implementation features a custom NumPy-based linear algebra pipeline for PCA and eigenfaces, facial image reconstruction from reduced representations, and face recognition evaluations using nearest-neighbor matching alongside traditional machine learning classifiers.

---

## 🎯 Objectives

* Load grayscale face images and map them into numerical feature vectors.
* Build reproducible, subject-stratified training and testing partitions where every subject is represented in both.
* Implement PCA from scratch using both **eigendecomposition** and **Singular Value Decomposition (SVD)**.
* Compare implementation outcomes via eigenvalues and principal directions.
* Determine the exact number of principal components required to retain $\ge 90\%$ of cumulative variance.
* Visualize eigenfaces and reconstruct facial images from low-dimensional projections.
* Evaluate dimensionality reduction via representation-size and reconstruction-error metrics.
* Compare raw-pixel versus PCA-based nearest-neighbor face recognition.
* Benchmark KNN, Logistic Regression, and SVM classifiers across varying principal component thresholds.

---

## 📊 Dataset & Train/Test Split

### Dataset Specifications

The **ORL / AT&T Face Database** contains grayscale facial images of 40 subjects with 10 variations per subject.

| Property | Value |
| --- | --- |
| **Number of Subjects** | 40 |
| **Images per Subject** | 10 |
| **Total Images** | 400 |
| **Image Dimensions** | $112 \times 92$ pixels |
| **Features per Image** | 10,304 |

> 📌 *Note: The dataset is excluded from public repositories. Download it from the [AT&T Laboratories Cambridge ORL Database](https://cam-orl.co.uk/facedatabase.html) and configure your local path in the notebook.*

### Partition Split (`seed=42`)

Each subject contributes 6 images to the training pool and 4 images to the testing pool.

| Partition | Images | Features / Image | Total Values |
| --- | --- | --- | --- |
| **Training Set** | 240 | 10,304 | 2,472,960 |
| **Testing Set** | 160 | 10,304 | 1,648,640 |
| **Total** | **400** | **10,304** | **4,121,600** |

---

## 🔬 Methodology & Implementation

### 1. Representation & PCA Pipeline

* **Preprocessing:** Images are flattened into 10,304-dimensional vectors and centered by subtracting the training mean face.
* **PCA Approaches:** Computed via Covariance-based Gram matrix eigendecomposition and NumPy SVD. Both yielded consistent principal directions (average absolute cosine similarity of the first 20 directions = $1.0000$).
* **Variance Retention:** **85 principal components** were selected to capture **$90.09\%$** of the cumulative variance (exceeding the $90\%$ threshold).

### 2. Dimensionality Reduction & Compression

* **Original Training Representation:** $2,472,960$ values ($240 \times 10,304$).
* **Reduced Representation:** $20,400$ values ($240 \times 85$).
* **Compression Ratio:** **121.22:1**
* **Representation Size Reduction:** **99.18%**
* **Reconstruction Error:** Average Mean Squared Error (MSE) across sample test images was **$426.99$**.

---

## 🏆 Results & Performance

### 1. Nearest-Neighbor Recognition

* **Raw-Pixel NN Accuracy:** **$96.88\%$** (155 / 160 correct)
* **PCA-Based NN Accuracy (85 components):** **$96.25\%$** (154 / 160 correct)
* *Takeaway:* Raw pixels achieved a marginal $+0.63\%$ accuracy lead, but PCA accomplished this with a $99.18\%$ reduction in feature dimensions.

### 2. Traditional Classifiers (at 85 Components)

| Classifier | Accuracy | Correct Predictions |
| --- | --- | --- |
| **PCA Nearest Neighbor** | 96.25% | 154 / 160 |
| **KNN ($k=1$)** | 96.25% | 154 / 160 |
| **Logistic Regression** | 96.88% | 155 / 160 |
| **SVM (RBF Kernel)** | **97.50%** | 156 / 160 |

### 3. Impact of Component Count on Classifier Accuracy

| PCA Components | KNN Accuracy | Logistic Regression Accuracy | SVM (RBF) Accuracy |
| --- | --- | --- | --- |
| **20** | 94.38% | 97.50% | 97.50% |
| **40** | 96.88% | 97.50% | **98.12%** |
| **60** | 96.88% | 97.50% | **98.12%** |
| **85** | 96.25% | 96.88% | 97.50% |
| **120** | 96.88% | **98.12%** | 97.50% |

> ⭐ **Peak Performance:** The highest test accuracy of **$98.12\%$** was achieved by **SVM with 40 or 60 components**, and **Logistic Regression with 120 components**, demonstrating that maximum components do not strictly correlate with peak generalization.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **NumPy:** Custom linear algebra operations, SVD, eigendecomposition, and projections.
* **Scikit-learn:** Classifier benchmarking (KNN, Logistic Regression, SVM).
* **Pandas:** Results aggregation and data structuring.
* **Matplotlib & Pillow (PIL):** Image processing, eigenface visualization, and reconstruction plotting.

---

## ⚙️ Installation & Execution

### Prerequisites

Install dependencies via `pip`:

```bash
pip install numpy pandas matplotlib Pillow scikit-learn jupyter

```

Alternatively, install directly from the requirements file:

```bash
pip install -r requirements.txt

```

### Running the Project

1. Extract the ORL/AT&T Face Database locally.
2. Open `202618053-Lab-07.ipynb` in Jupyter Notebook or VS Code.
3. Update the dataset path variable to match your local setup:
```python
from pathlib import Path
dataset_path = Path("./archive")

```


4. Execute all notebook cells sequentially.

---

## 🔍 Limitations & Future Scope

* **Single Split Evaluation:** Performance metrics rely on a single reproducible random split (`seed=42`). Implementing k-fold cross-validation would provide a more robust variance estimate.
* **Environmental Sensitivity:** Pixel-level PCA remains sensitive to variations in lighting, facial expressions, and alignment.
* **Storage Accounting:** Compression ratios measure representation-size reduction of features and exclude the overhead of storing the mean face and principal-component matrix.

---

## 📌 Course Details

* **Course:** DS605 — Fundamentals of Machine Learning
* **Assignment:** Lab 7 — Image Representation, PCA, Eigenfaces and Face Recognition
* **Student ID:** 202618053
* **Name:** Dhruv Bhanushali