import pandas as pd

df = pd.read_csv('202618053-Lab-1/raw_books2.csv')

total_records = len(df)
duplicate_upcs = df['upc'].duplicated().sum()
missing_values = df.isnull().sum()

print(f"Total Records Scraped: {total_records}")
print(f"Duplicate UPCs Count: {duplicate_upcs}")
print("\nMissing Values per column:")
print(missing_values)