import pandas as pd
import re

def clean_data():
    df = pd.read_csv('202618053-Lab-1/raw_books2.csv')
    print(f"Loaded {len(df)} raw records.")

    df['title'] = df['title'].astype(str).str.strip()
    df['category'] = df['category'].astype(str).str.strip()
    df['description'] = df['description'].astype(str).str.strip()
    df['upc'] = df['upc'].astype(str).str.strip()

    df = df.drop_duplicates(subset=['upc']).reset_index(drop=True)

    df['price'] = df['price'].astype(str).str.extract(r'(\d+\.\d+)')[0].astype(float)

    rating_map = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}
    df['rating'] = df['rating'].astype(str).str.strip().map(rating_map)

    def extract_stock(val):
        match = re.search(r'\((\d+)\s*available\)', str(val))
        if match:
            return int(match.group(1))
        elif 'In stock' in str(val):
            return 1
        return 0

    df['stock_count'] = df['availability'].apply(extract_stock)

    df['description_word_count'] = df['description'].apply(lambda x: len(str(x).split()))

    df['price_band'] = pd.qcut(df['price'], q=3, labels=['Budget', 'Moderate', 'Expensive'])

    df['value_score'] = (df['rating'] / df['price'] * 100).round(2)

    clean_cols = [
        'title', 'category', 'price', 'rating', 'stock_count',
        'description', 'upc', 'number_of_reviews', 'product_url',
        'description_word_count', 'price_band', 'value_score'
    ]
    
    df_clean = df[clean_cols]

    df_clean.to_csv('cleaned_books.csv', index=False)
    print("✅ Preprocessing complete! Exported to 'cleaned_books.csv'.")
    print(df_clean[['title', 'price', 'rating', 'stock_count', 'price_band', 'value_score']].head())

if __name__ == "__main__":
    clean_data()