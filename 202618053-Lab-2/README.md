
# DS605: Fundamentals of Machine Learning
## Lab Assignment 2 – Vectorized Programming with NumPy & Data Wrangling with Pandas

**Course:** DS605 - Fundamentals of Machine Learning  
**Student Name:** Dhruv Bhanushali  
**Student ID:** 202618059 
**Dataset (Part B):** Kaggle Titanic Dataset (`train.csv`)  

---

## 📌 Project Overview

This repository contains the complete implementation for **Lab Assignment 2**, covering:
1. **Part A — Vectorized Operations with NumPy**: Multi-dimensional array manipulations, summary statistics, linear algebra (matrix multiplication, transpose, determinant, inverse verification), and normal distribution sampling with plot generation.
2. **Part B — Data Wrangling & Analysis with Pandas**: Exploratory Data Analysis (EDA) on the Titanic dataset, including complex query filtering, missing value imputations (mean, median, mode, random sampling), outlier detection using IQR bounds, feature engineering (`FamilySize`, `IsAlone`), pivot tables, dataset exporting (`cleaned_titanic.csv`), and plot exports into the `assets/` directory.

---

## 🚀 Tasks & Implementation Highlights

### Part A: Vectorized Programming with NumPy

* **Task 1: Arrays, Statistics, and Indexing**
  * Generated 100 random integers with a fixed seed (`np.random.seed(42)`).
  * Calculated summary statistics: `min`, `max`, `median`, `mean`, `std`.
  * Demonstrations of `np.arange()`, `np.zeros()`, `np.ones()`, `np.linspace()`, 2D/3D array indexing, slicing, reshaping, and flattening.
  * **Difference Note (`linspace` vs `arange`):** `np.arange(start, stop, step)` generates values within a given interval using a fixed step size, whereas `np.linspace(start, stop, num)` generates a specified number of evenly spaced samples over a specified interval.

* **Task 2: Vectorized Arithmetic & Linear Algebra**
  * Matrix operations performed completely without explicit Python loops using vectorized NumPy operations.
  * Verified matrix invertibility ($A \cdot A^{-1} \approx I$) using `np.allclose()`.

* **Task 3: Normal Distribution & Histogram**
  * Generated 1,000 values from $\mathcal{N}(\mu=50, \sigma=15)$.
  * Calculated sample mean and sample standard deviation, verifying them against the target parameters.
  * Saved histogram to `assets/normal_distribution.png`.

---

### Part B: Data Wrangling with Pandas (Titanic Dataset)

* **Task 4: Inspection & Selection**
  * Inspected data using `.head()`, `.tail()`, `.shape`, `.columns`, `.info()`, and `.describe()`.
  * **Difference Note (`loc` vs `iloc`):** 
    * `loc` is **label-based** (indexing by row/column names; slices are inclusive of endpoints).
    * `iloc` is **integer position-based** (indexing by numeric 0-based positions; slices follow standard Python exclusive bounds).

* **Task 5: Filtering & Querying**
  * *Male passengers older than 50:* Filtered via Boolean indexing.
  * *Female first-class passengers:* Calculated total count and survival percentage.
  * *Age 20–40, Fare > overall median, and survived:* Executed compound Boolean indexing.
  * *Solo travelers (SibSp=0, Parch=0), age < 30, non-survivors:* Executed condition filtering.
  * *Embarked='S', Pclass 2 or 3, Fare > Southampton median:* Filtered using Pandas `.query()`.

* **Task 6: Grouping & Aggregations**
  * Aggregated survival rates, average age, and average fare grouped across `Sex`, `Pclass`, and `Embarked` ports.

* **Task 7: Missing Values & Outlier Analysis**
  * Calculated missing value counts and percentages for all columns.
  * Visualized missing column counts via bar plot (`assets/missing_values.png`).
  * Evaluated 4 imputation strategies for `Age`: Mean, Median, Mode, and Random Value Sampling, imputing missing values with mean `Age`.
  * Detected `Fare` outliers using the Interquartile Range (IQR) method ($1.5 \times \text{IQR}$).

* **Task 8: Feature Engineering & Pivot Table**
  * Engineered new features:
    $$\text{FamilySize} = \text{SibSp} + \text{Parch} + 1$$
    $$\text{IsAlone} = \begin{cases} 1 & \text{if } \text{FamilySize} = 1 \\ 0 & \text{otherwise} \end{cases}$$
  * Constructed pivot table: `Sex` vs `Pclass` aggregated by mean `Survived`. Identified demographic groups with highest and lowest survival rates.
  * Exported the fully processed dataset to `cleaned_titanic.csv`.

* **Task 9: Visualizations**
  * Generated and saved visualization assets:
    1. `assets/correlation_heatmap.png`: Correlation matrix across numerical columns.
    2. `assets/survival_by_sex.png`: Bar plot of male vs. female survival rates.
    3. `assets/age_vs_fare.png`: Scatter plot comparing age vs fare, hue-coded by survival outcome.

---

## 📊 Key Observations & Insights

1. **Strong Gender Disparity in Survival:** Female passengers had a drastically higher overall survival rate (~74.2%) compared to male passengers (~18.9%), reflecting the strict implementation of "women and children first" evacuation protocols.
2. **Socioeconomic Class Influence (`Pclass`):** Passenger class shows a strong inverse relationship with survival probability. 1st Class passengers experienced the highest survival rate (~62.9%), followed by 2nd Class (~47.3%), and 3rd Class (~24.2%).
3. **Combined Demographics (Pivot Analysis):** The demographic subgroup with the highest survival rate was **1st Class Females** (>96% survival), whereas **3rd Class Males** had the lowest overall survival rate (<14% survival).
4. **Impact of Ticket Fare:** `Fare` exhibits a positive correlation with survival rate. Outlier analysis identified fares above ~\$66.30 as upper bound outliers; passengers in these high-fare tiers were predominantly 1st Class and showed significantly higher survival rates.
5. **Port of Embarkation Effects:** Passengers embarking from Cherbourg (`Embarked = 'C'`) had the highest average fare and highest survival rate (~55.4%) compared to those embarking from Southampton (`'S'`, ~33.7%) or Queenstown (`'Q'`, ~38.9%).
6. **Family Size Dynamics:** Passengers traveling alone (`IsAlone = 1`) had a noticeably lower survival rate (~30.4%) compared to those traveling with small families ($\text{FamilySize} \in [2, 4]$). However, very large families ($\text{FamilySize} \ge 5$) saw survival rates drop sharply due to evacuation constraints.
7. **Missing Data Patterns:** Missingness was heavily concentrated in `Cabin` (~77% missing data) and `Age` (~20% missing data). Imputing `Age` with the mean preserved sample central tendency without altering overall group distributions.

---

## 🛠️ Instructions to Run the Code

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
   cd YOUR_REPOSITORY_NAME

```

2. **Install required dependencies:**
```bash
pip install numpy pandas matplotlib seaborn jupyter

```


3. **Launch Notebook:**
```bash
jupyter notebook Lab2_NumPy_Pandas.ipynb

```


4. **Execution:**
Executing the notebook cells sequentially generates all numerical outputs, exports the preprocessed dataset to `cleaned_titanic.csv`, and saves all visual plots into the `assets/` directory.

```

