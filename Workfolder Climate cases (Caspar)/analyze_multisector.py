import pandas as pd
from collections import Counter

# Read the CSV file
df = pd.read_csv('combined_climate_data.csv')

# Define sector columns
sector_columns = [
    'agriculture', 'construction', 'education', 'energy_supply',
    'enviromental_management', 'extractive', 'finance_insurance',
    'fishing', 'forestry', 'informal', 'manufacturing', 'media',
    'public', 'trade', 'transportation_logistics', 'water_management'
]

# 1. Count mentions of each category
print("=" * 80)
print("INDIVIDUAL SECTOR COUNTS")
print("=" * 80)
sector_counts = {}
for col in sector_columns:
    count = df[col].sum()
    sector_counts[col] = int(count)

# Sort by count (descending)
sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
for sector, count in sorted_sectors:
    print(f"{sector:30s}: {count:5d}")

print(f"\nTotal cases: {len(df)}")

# 2. Find the most common combinations
print("\n" + "=" * 80)
print("TOP 20 SECTOR COMBINATIONS")
print("=" * 80)

# Create combination strings for each row
combinations = []
for idx, row in df.iterrows():
    active_sectors = [col for col in sector_columns if row[col] == 1]
    if active_sectors:
        # Sort to ensure consistent representation
        combo = tuple(sorted(active_sectors))
        combinations.append(combo)
    else:
        combinations.append(('none',))

# Count combinations
combo_counter = Counter(combinations)

# Get top 20
top_20 = combo_counter.most_common(20)

for i, (combo, count) in enumerate(top_20, 1):
    if combo == ('none',):
        combo_str = "No sectors"
    else:
        combo_str = ", ".join(combo)
    print(f"\n{i:2d}. Count: {count:4d}")
    print(f"    Sectors: {combo_str}")

# Additional statistics
print("\n" + "=" * 80)
print("ADDITIONAL STATISTICS")
print("=" * 80)

# Calculate how many sectors per case
df['num_sectors'] = df[sector_columns].sum(axis=1)
print(f"Average sectors per case: {df['num_sectors'].mean():.2f}")
print(f"Median sectors per case: {df['num_sectors'].median():.0f}")
print(f"Max sectors in a single case: {df['num_sectors'].max():.0f}")
print(f"Min sectors in a single case: {df['num_sectors'].min():.0f}")

# Distribution of number of sectors
print("\nDistribution of number of sectors per case:")
sector_dist = df['num_sectors'].value_counts().sort_index()
for num, count in sector_dist.items():
    print(f"  {int(num):2d} sectors: {count:4d} cases")
