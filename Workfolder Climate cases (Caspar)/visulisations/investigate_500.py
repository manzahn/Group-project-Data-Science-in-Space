import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('combined_climate_data.csv')

# Define sector columns
sector_columns = [
    'agriculture', 'construction', 'education', 'energy_supply',
    'enviromental_management', 'extractive', 'finance_insurance',
    'fishing', 'forestry', 'informal', 'manufacturing', 'media',
    'public', 'trade', 'transportation_logistics', 'water_management'
]

print("=" * 80)
print("INVESTIGATING THE SUSPICIOUS 500s")
print("=" * 80)
print(f"\nTotal rows in dataset: {len(df)}")

# Count each sector
print("\nSector counts:")
for col in sector_columns:
    count = df[col].sum()
    print(f"  {col:30s}: {int(count)}")

# Check for duplicate rows
print(f"\n\nChecking for duplicates:")
print(f"Total rows: {len(df)}")
print(f"Unique rows: {len(df.drop_duplicates())}")
print(f"Duplicate rows: {len(df) - len(df.drop_duplicates())}")

# Check if there's a pattern - are certain rows identical across sectors?
# Let's look at the sector columns only
sector_data = df[sector_columns]
print(f"\nUnique sector combinations: {len(sector_data.drop_duplicates())}")

# Check first 20 rows to see pattern
print("\n\nFirst 20 rows - sector values:")
print(df[sector_columns].head(20).to_string())

# Check if certain cases have all 1s (or patterns)
df['sum_sectors'] = df[sector_columns].sum(axis=1)
print("\n\nDistribution of total sectors per case:")
print(df['sum_sectors'].value_counts().sort_index())

# Look for rows with exactly 15 sectors (all except informal)
rows_with_15 = df[df['sum_sectors'] == 15]
print(f"\n\nRows with exactly 15 sectors: {len(rows_with_15)}")
if len(rows_with_15) > 0:
    print("Sample of these rows:")
    print(rows_with_15[['Case URL', 'Case Filing Year for Action'] + sector_columns].head(10).to_string())

# Check if there are exactly 500 rows with each sector = 1 in some systematic way
print("\n\n" + "=" * 80)
print("DETAILED INVESTIGATION")
print("=" * 80)

# For each sector, show distribution of values
print("\nValue distributions (should only be 0 and 1):")
for col in sector_columns:
    unique_vals = df[col].unique()
    print(f"  {col}: {sorted(unique_vals)}")

# Check if maybe there's some artificial balancing
# Let's see if there's a maximum cap of 500 per sector
print("\n\nTheory: Could rows be duplicated or synthetically generated?")
print("Let's check the Case URL duplicates:")
url_counts = df['Case URL'].value_counts()
duplicated_urls = url_counts[url_counts > 1]
print(f"Number of duplicate Case URLs: {len(duplicated_urls)}")
if len(duplicated_urls) > 0:
    print(f"\nMost duplicated URLs:")
    print(duplicated_urls.head(10))

# Check if there's a systematic issue
# Calculate correlation between sectors
print("\n\nCorrelation between sectors:")
correlation = df[sector_columns].corr()
print("\nSectors with high correlation (>0.5):")
for i in range(len(sector_columns)):
    for j in range(i+1, len(sector_columns)):
        corr_val = correlation.iloc[i, j]
        if abs(corr_val) > 0.5:
            print(f"  {sector_columns[i]:30s} <-> {sector_columns[j]:30s}: {corr_val:.3f}")
