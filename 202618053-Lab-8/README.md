# DS605 Lab 8: Decision Trees and Support Vector Machines

**Student Name:** Diya Tilwani  
**Course:** DS605 — Machine Learning  
**Lab Assignment:** Lab 8 — Decision Trees and Support Vector Machines

---

## 1. Introduction

The purpose of this laboratory exercise is to study Decision Tree and Support Vector Machine (SVM) classifiers through both manual implementations and Scikit-learn. The lab focuses on understanding the working principles of these algorithms and comparing their performance on classification tasks.

The experiments involve implementing impurity measures, constructing a Decision Tree recursively, optimizing a linear SVM using hinge loss, and examining different SVM kernels. Model performance is evaluated using classification metrics and execution time.

## 2. Objectives

The main objectives of this lab are:

- Implement Gini impurity and entropy using Python and NumPy.
- Develop a Decision Tree classifier without using Scikit-learn's tree-building algorithm.
- Implement a linear soft-margin SVM using subgradient descent.
- Compare manually implemented algorithms with their Scikit-learn counterparts.
- Study the influence of hyperparameters such as tree depth, regularization strength, learning rate, and iteration count.
- Compare linear, polynomial, and RBF kernels on nonlinear data.
- Visualize decision boundaries and tree structures.
- Evaluate models using accuracy, precision, recall, F1-score, training time, and prediction time.

## 3. Dataset Description

### 3.1 Banknote Authentication Dataset

The Banknote Authentication dataset is used to classify banknotes as genuine or forged based on numerical features extracted from their images.

| Property | Details |
|---|---|
| Total observations | 1,372 |
| Number of input features | 4 |
| Target column | `class` |
| Classification type | Binary |
| Missing values | None in the inspected dataset |
| Class 0 | 762 observations |
| Class 1 | 610 observations |

The dataset contains the following input attributes:

| Feature | Description |
|---|---|
| `variance` | Variance of the wavelet-transformed image |
| `skewness` | Skewness of the wavelet-transformed image |
| `curtosis` | Curtosis of the wavelet-transformed image |
| `entropy` | Entropy of the image |

Class 0 represents 55.54% of the observations, while class 1 accounts for 44.46%. The distribution is reasonably balanced, making the dataset suitable for binary classification experiments.

### 3.2 Initial Data Analysis

Before training the models, the dataset was examined to understand its structure and characteristics. The analysis included:

- Inspecting the first five rows and dataset dimensions.
- Checking column data types and missing values.
- Examining the distribution of the target classes.
- Calculating descriptive statistics such as mean, standard deviation, quartiles, minimum, and maximum.
- Plotting selected features against the target classes.

The input features have different scales. For instance, the standard deviation of `skewness` is approximately 5.869, whereas that of `entropy` is approximately 2.101. Feature standardization was therefore applied in the SVM experiments.

## 4. Part A — Decision Tree Implementation

### 4.1 Implementation Approach

A Decision Tree classifier was developed from scratch using Python and NumPy. Instead of relying on Scikit-learn's tree-building procedure, the manual implementation determines splits and constructs the tree recursively.

The implementation follows these steps:

1. Calculate Gini impurity and entropy for a set of class labels.
2. Generate candidate thresholds for each input feature.
3. Divide the observations into left and right groups using each threshold.
4. Calculate the weighted impurity after each split.
5. Select the split that produces the greatest impurity reduction.
6. Repeat the splitting process recursively for the resulting child nodes.
7. Stop when a node is pure, the maximum depth is reached, the minimum sample condition is met, or no useful split is available.
8. Predict the class of individual observations and batches of observations.

The manually constructed tree was compared with Scikit-learn's `DecisionTreeClassifier` using both Gini impurity and entropy as splitting criteria.

### 4.2 Effect of Maximum Tree Depth

Different maximum-depth settings were tested to investigate their influence on classification accuracy. The experiments used depths of 1, 2, 3, 5, 10, and unlimited depth (`None`).

| Criterion | Maximum depth | Manual accuracy | Scikit-learn accuracy |
|---|---:|---:|---:|
| Entropy | 1 | 82.91% | 82.91% |
| Entropy | 2 | 87.64% | 87.64% |
| Entropy | 3 | 90.91% | 90.91% |
| Entropy | 5 | 93.45% | 93.45% |
| Entropy | 10 | 93.45% | 93.09% |
| Entropy | None | 93.45% | 93.45% |
| Gini | 1 | 82.91% | 82.91% |
| Gini | 2 | 87.64% | 87.64% |
| Gini | 3 | 90.91% | 90.91% |
| Gini | 5 | 93.09% | 93.09% |
| Gini | 10 | 92.00% | 92.36% |
| Gini | None | 92.00% | 92.00% |

### 4.3 Discussion

The best test accuracy obtained in these experiments was **93.45%**, achieved by the manual entropy-based tree at depths 5, 10, and unlimited depth. The Scikit-learn entropy tree also obtained 93.45% at depths 5 and unlimited depth.

Accuracy improved as the tree became deeper during the initial experiments. However, increasing the depth beyond a certain point did not consistently produce better results.

The manual implementation and Scikit-learn generally produced similar accuracy values. This indicates that the manually implemented splitting procedure behaved comparably to the standard implementation for the tested configurations.

The manual tree required considerably more training time because its splitting operations were implemented directly in Python and NumPy, whereas Scikit-learn uses optimized internal routines.

## 5. Part B — Linear SVM from Scratch

### 5.1 Implementation Approach

A linear soft-margin SVM was implemented using NumPy. The manual model uses gradient-based optimization rather than Scikit-learn's SVM optimization algorithm.

The implementation consists of the following operations:

1. Standardize the features using statistics calculated from the training set.
2. Convert the target labels from `{0, 1}` to `{-1, +1}`.
3. Calculate the hinge loss for the training observations.
4. Compute subgradients for the weight vector and bias.
5. Update the parameters using the selected learning rate.
6. Generate predictions using the learned decision function.
7. Plot the separating hyperplane, margin boundaries, and margin-violating observations.

### 5.2 Objective Function

The manual SVM minimizes the following objective:

\[
J(w,b)=\frac{1}{2}\|w\|^2+
\frac{C}{n}\sum_{i=1}^{n}
\max\left(0,1-y_i(w^\top x_i+b)\right)
\]

The objective has two main components:

- **Regularization:** The term \(\frac{1}{2}\|w\|^2\) discourages excessively large weights and encourages a wider margin.
- **Hinge loss:** The second term penalizes observations that are incorrectly classified or fall within the required margin.

Here, \(w\) represents the weight vector, \(b\) is the bias, \(C\) controls the penalty associated with hinge loss, \(n\) is the number of training observations, \(x_i\) is an input feature vector, and \(y_i\) is its class label in \(\{-1,+1\}\).

### 5.3 Hyperparameter Investigation

The manual SVM was tested with different regularization values, learning rates, and iteration counts.

The experiments demonstrated that the value of \(C\) determines the balance between the regularization penalty and classification errors. The learning rate affects the stability and speed of optimization, while the number of iterations determines how many parameter updates are performed.

A larger iteration count does not automatically guarantee better test accuracy. The result also depends on the learning rate, regularization strength, and convergence behavior.

The manual implementation helps illustrate the optimization process, while Scikit-learn provides a more optimized implementation. Their results must be interpreted separately because the optimization procedures and hyperparameter settings can differ.

## 6. Part C — Comparison of SVM Kernels

### 6.1 Synthetic Dataset

The `make_moons` dataset was generated using Scikit-learn to investigate how different SVM kernels handle nonlinear class boundaries.

The dataset contained 600 observations, a noise level of 0.20, and a random state of 42. An 80:20 train-test split was used. Standardization was fitted on the training data and then applied to both the training and testing sets.

Three kernel configurations were examined:

- Linear kernel
- Polynomial kernel with degree 3
- Radial Basis Function (RBF) kernel

### 6.2 Experimental Results

| Kernel | Accuracy | Precision | Recall | F1-score | Training time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|
| RBF | 95.00% | 92.19% | 98.33% | 95.16% | 0.001602 | 0.000558 |
| Linear | 88.33% | 87.10% | 90.00% | 88.52% | 0.003826 | 0.000507 |
| Polynomial | 88.33% | 91.07% | 85.00% | 87.93% | 0.001321 | 0.000274 |

### 6.3 Interpretation

The RBF kernel achieved the highest accuracy of **95.00%** and the highest F1-score of **95.16%**. Its recall of 98.33% indicates that it correctly identified most of the positive-class observations.

The linear kernel achieved 88.33% accuracy. Its performance was lower because a straight decision boundary cannot adequately represent all the curved patterns in the dataset.

The polynomial kernel also achieved 88.33% accuracy. It produced a higher precision value than the linear kernel but a lower recall.

Among the three models, the polynomial kernel had the shortest measured training time and prediction time in this experiment. These timings depend on the execution environment and may vary across runs.

Overall, the results demonstrate that nonlinear kernels can provide better classification performance when the relationship between features and class labels cannot be represented adequately by a straight boundary.

## 7. Part D — Overall Model Evaluation

### 7.1 Experimental Setup

The final evaluation compared five models on the Banknote Authentication dataset using all four input features:

- Manual Decision Tree
- Scikit-learn Decision Tree
- Manual Linear SVM
- Scikit-learn Linear SVM
- Scikit-learn RBF SVM

The same train-test split established earlier in the notebook was used for the comparison. Standardized features were supplied to the SVM models, with the scaler fitted only on the training data to prevent test-data leakage.

### 7.2 Final Results

| Model | Accuracy | Precision | Recall | F1-score | Training time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Scikit-learn RBF SVM | 100.00% | 100.00% | 100.00% | 100.00% | 0.00362 | 0.00102 |
| Manual Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 0.63822 | 0.00022 |
| Scikit-learn Decision Tree | 99.27% | 98.39% | 100.00% | 99.19% | 0.00188 | 0.00056 |
| Scikit-learn Linear SVM | 97.45% | 94.57% | 100.00% | 97.21% | 0.00488 | 0.00088 |
| Manual Linear SVM | 88.36% | 94.12% | 78.69% | 85.71% | 0.15154 | 0.00002 |

*Note: Training and prediction times are recorded in seconds and correspond to the particular execution run and environment.*

### 7.3 Analysis of the Results

The Scikit-learn RBF SVM delivered the strongest result, achieving **100.00% accuracy, precision, recall, and F1-score** on the selected test split.

Both Decision Tree implementations achieved 99.27% accuracy and identical reported precision, recall, and F1-score. This close agreement supports the correctness of the manual implementation for the evaluated configuration.

The Scikit-learn linear SVM obtained 97.45% accuracy, while the manual linear SVM achieved 88.36%. The manual model's recall of 78.69% indicates that it failed to identify a larger proportion of positive-class observations.

The execution-time comparison also revealed a substantial difference between the manual and library-based implementations. The manual Decision Tree took 0.63822 seconds to train, compared with 0.00188 seconds for the Scikit-learn Decision Tree.

The perfect test score of the RBF SVM applies only to the selected split. It does not establish that the model will classify every future observation correctly. Cross-validation or evaluation on an independent dataset would provide stronger evidence of its generalization performance.

## 8. Overall Findings

| Experiment | Key observation |
|---|---|
| Manual Decision Tree | Impurity-based splitting and recursive tree construction were implemented successfully. |
| Decision Tree comparison | Manual and Scikit-learn models generally produced similar classification accuracy. |
| Manual Linear SVM | The implementation demonstrated hinge-loss optimization and the effects of hyperparameter choices. |
| Kernel comparison | The RBF kernel performed best on the nonlinear `make_moons` dataset. |
| Final Banknote evaluation | The Scikit-learn RBF SVM achieved the highest observed test accuracy. |

The experiments highlight the importance of choosing suitable models, tuning hyperparameters, standardizing features where appropriate, and evaluating performance with multiple metrics. They also demonstrate that a from-scratch implementation is valuable for understanding an algorithm, even when an optimized library implementation is more computationally efficient.

## 9. Software and Libraries

The laboratory work was carried out using the following technologies:

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 10. Instructions to Execute the Notebook

### Step 1: Clone the repository

```bash
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with the public GitHub repository URL and the corresponding folder name.

### Step 2: Install the required packages

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Step 3: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 4: Execute the notebook

Open the Lab 8 notebook and run the cells in sequence, beginning with dataset loading and exploratory analysis and continuing through the final model comparison.

The notebook obtains the Banknote Authentication dataset from its configured public source. An internet connection may be necessary if the dataset has not already been downloaded.

## 11. Conclusion

This laboratory exercise provided a practical study of Decision Trees and Support Vector Machines through manual implementations and Scikit-learn models.

The Decision Tree implementation demonstrated how Gini impurity, entropy, impurity reduction, and recursive splitting can be combined to construct a classifier. The linear SVM implementation provided practical insight into hinge loss, regularization, and subgradient-based parameter updates.

The kernel experiments showed that the RBF kernel was more effective than the linear and polynomial kernels on the selected nonlinear dataset. In the final Banknote Authentication comparison, the Scikit-learn RBF SVM achieved 100.00% test accuracy, while both Decision Tree implementations achieved 99.27%.

The results also showed that optimized library implementations can significantly reduce training time. Overall, the lab strengthened the understanding of classification algorithms, optimization, model evaluation, and the trade-offs between implementing algorithms manually and using established machine-learning libraries.