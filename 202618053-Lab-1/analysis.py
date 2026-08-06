import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('202618053-Lab-1/cleaned_books.csv')


print("=== Task 1: Scraping Statistics ===")
print(f"Total Scraped Records: {len(df)}")
print(f"Duplicate UPCs: {df['upc'].duplicated().sum()}")
print("Missing Values:\n", df.isnull().sum())
print("="*40)


plt.figure()
sns.histplot(df['price'], kde=True, color='skyblue', bins=15)
plt.title('1. Price Distribution of Scraped Books')
plt.xlabel('Price (£)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('price_distribution.png')
plt.close()

plt.figure()
sns.countplot(x='rating', data=df, palette='viridis')
plt.title('2. Rating Distribution (1 to 5 Stars)')
plt.xlabel('Rating (Stars)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('rating_distribution.png')
plt.close()

plt.figure(figsize=(10, 8))
avg_price_cat = df.groupby('category')['price'].mean().sort_values(ascending=False).reset_index()
sns.barplot(x='price', y='category', data=avg_price_cat, palette='magma')
plt.title('3. Average Price by Category')
plt.xlabel('Average Price (£)')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('avg_price_by_category.png')
plt.close()

plt.figure()
sns.boxplot(x='rating', y='price', data=df, palette='Set2')
sns.stripplot(x='rating', y='price', data=df, color='black', alpha=0.5, jitter=0.2)
plt.title('4. Relationship: Price vs Rating Distribution')
plt.xlabel('Rating (Stars)')
plt.ylabel('Price (£)')
plt.tight_layout()
plt.savefig('price_vs_rating.png')
plt.close()

text = " ".join(df['description'].dropna().tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='tab10').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('5. Word Cloud of Book Descriptions')
plt.tight_layout()
plt.savefig('wordcloud.png')
plt.close()


print("\n" + "="*50)
print("          TASK 4: INSIGHTS & REPORT")
print("="*50)

# 1. Price Stats
print("\n1. PRICE & RATING SUMMARY:")
print(f"   - Average Price: £{df['price'].mean():.2f}")
print(f"   - Price Range  : £{df['price'].min():.2f} to £{df['price'].max():.2f}")
print(f"   - Price vs Rating Correlation: {df['price'].corr(df['rating']):.4f}")

# 2. Top Categories
print("\n2. TOP 3 MOST REPRESENTED CATEGORIES:")
top_cats = df['category'].value_counts().head(3)
for cat, count in top_cats.items():
    print(f"   - {cat}: {count} books")

# 3. Category Pricing Extremes
cat_means = df.groupby('category')['price'].mean()
print("\n3. CATEGORY PRICING EXTREMES:")
print(f"   - Most Expensive Category : {cat_means.idxmax()} (£{cat_means.max():.2f} avg)")
print(f"   - Most Affordable Category: {cat_means.idxmin()} (£{cat_means.min():.2f} avg)")

# 4. Top Value Books
print("\n4. TOP 3 BEST VALUE BOOKS (Rating / Price * 100):")
top_value = df.sort_values(by='value_score', ascending=False)[['title', 'price', 'rating', 'value_score']].head(3)
for idx, row in top_value.iterrows():
    print(f"   - '{row['title']}' | {row['rating']} Stars | £{row['price']} | Score: {row['value_score']}")

print("="*50)