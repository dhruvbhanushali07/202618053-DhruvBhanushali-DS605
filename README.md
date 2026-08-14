
# DS-605: Lab Assignment 1 - Book Data Scraping & Analysis Pipeline

**Name:** Dhruv Bhanushali  
**Student ID:** 202618053

**Target Website:** [Books to Scrape](https://books.toscrape.com)

---

##  Project Overview
This repository contains a complete end-to-end data pipeline built using Python and Scrapy. The pipeline scrapes book details across catalog pages, cleans and transforms raw fields, engineers insightful domain metrics, generates data visualizations, and reports key analytical takeaways.

---

##  Repository Structure
```text
202618053-Lab-1/
├── books_spider.py          # Task 1: Scrapy spider for catalog and detail page extraction
├── raw_books2.csv           # Task 1: Output raw dataset from web crawler
├── preprocess.py            # Task 2: Data cleaning & feature engineering script
├── cleaned_books.csv        # Task 2: Output preprocessed & clean dataset
├── analysis.py              # Task 3 & 4: Data visualization and analysis script
├── price_distribution.png   # Task 3: Distribution plot of book prices
├── rating_distribution.png  # Task 3: Count plot of book star ratings
├── avg_price_by_category.png# Task 3: Horizontal bar chart of average category prices
├── price_vs_rating.png      # Task 3: Box & strip plot comparing price across ratings
├── wordcloud.png            # Task 3: Word cloud generated from book descriptions
└── README.md                # Task 4: Project documentation & detailed findings

```

---

##  Task 1: Data Scraping Summary

* **Total Records Scraped:** 100 books (across 5 catalog pages)
* **Duplicate UPC Count:** 0 (all UPCs are unique)
* **Missing Values:** 0 missing values across all required fields (`title`, `category`, `price`, `rating`, `availability`, `description`, `upc`, `num_reviews`, `product_url`)

---

##  Task 2: Data Preprocessing & Feature Engineering

The raw data was processed using Pandas (`preprocess.py`):

1. **Text Cleaning:** Stripped leading/trailing whitespace and linebreaks from titles, categories, and descriptions.
2. **Type Conversions:**
* Stripped currency characters (`£`) from prices and cast them to `float`.
* Mapped textual ratings (`One` to `Five`) to integer values (`1` to `5`).
* Extracted exact available stock integers from raw availability strings.


3. **Engineered Features:**
* `description_word_count`: Total word count in each product description.
* `price_band`: Binned books into `Budget`, `Moderate`, or `Expensive` using quantile boundaries.
* `value_score`: Score calculating value for money $(\frac{\text{Rating}}{\text{Price}} \times 100)$.



---

##  Task 3: Visualizations & Plots

### 1. Price Distribution

### 2. Rating Distribution

### 3. Average Price by Category

### 4. Price vs. Rating Relationship

### 5. Word Cloud of Book Descriptions

---

##  Task 4: Key Insights & Data-Driven Observations

1. **Price Range & Bimodal Distribution:**
* Book prices range from **£10.16** (*Patience*) to **£58.11** (*The Red Tent*), with a mean price of **£34.56** and a median of **£34.78**. The price distribution displays a bimodal structure with peaks near £18 and £52.


2. **No Linear Relationship Between Price and Rating:**
* The correlation coefficient between `price` and `rating` is **-0.1217**. Price does not dictate star rating quality on this platform—5-star ratings occur evenly across low-cost and premium price points.


3. **Category Representation:**
* **Sequential Art** (14 titles) and **Nonfiction** (12 titles) are the most heavily represented genres in the dataset, together making up over 25% of total scraped items.


4. **Category Pricing Extremes:**
* **Historical Fiction** (£53.74 avg) and **Politics** (£51.33 avg) are the most expensive categories on average, whereas **Spirituality** (£25.09 avg) and **Young Adult** (£25.05 avg) are the most affordable.


5. **Top Value-for-Money Books:**
* Evaluated using our `value_score` metric, the top 3 best-value books are:
1. *Princess Between Worlds* (5 Stars | £13.34 | Score: 37.48)
2. *Princess Jellyfish 2-in-1 Omnibus, Vol. 01* (5 Stars | £13.61 | Score: 36.74)
3. *Sophie's World* (5 Stars | £15.94 | Score: 31.37)




6. **Description Themes:**
* The word cloud shows a dominant narrative focus on terms such as **"life"**, **"world"**, **"story"**, **"time"**, and **"new"**, reflecting character journeys and personal growth across genres.



---

##  Limitations of the Dataset & Analysis

* **Sandbox Dataset:** `books.toscrape.com` is a mock platform with randomized prices and ratings, explaining the lack of realistic correlation between product cost and ratings.
* **Absence of Review Text:** The target site lacks written customer reviews, limiting text analysis strictly to publisher descriptions rather than genuine customer sentiment.
* **Sample Constraints:** Scope was capped at 100 books, leaving smaller categories with only 1–2 sample data points.

---

##  How to Run the Pipeline

1. **Activate Virtual Environment:**
```bash
source .venv/bin/activate

```


2. **Run Scrapy Spider (Task 1):**
```bash
scrapy crawl books -o raw_books2.csv

```


3. **Run Data Preprocessing (Task 2):**
```bash
python preprocess.py

```


4. **Run Visualizations and Analysis (Task 3 & 4):**
```bash
python analysis.py

```
