import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('combined_climate_data.csv')

sector_columns = [
    'agriculture', 'construction', 'education', 'energy_supply',
    'enviromental_management', 'extractive', 'finance_insurance',
    'fishing', 'forestry', 'informal', 'manufacturing', 'media',
    'public', 'trade', 'transportation_logistics', 'water_management'
]

print("=" * 80)
print("DEEP INVESTIGATION: Is this data synthetic?")
print("=" * 80)

# Theory: If we have exactly 500 per sector, how is this mathematically possible?
# Let's work backwards

total_cases = len(df)
print(f"\nTotal cases: {total_cases}")

# Calculate expected distribution if random
print("\n" + "=" * 80)
print("MATHEMATICAL ANALYSIS")
print("=" * 80)

# If we have 500 for each of 15 sectors, that's 7500 sector assignments
# Plus 165 for informal = 7665 total
# Across 1543 cases, that's 7665/1543 = 4.97 sectors per case on average

total_assignments = sum(df[col].sum() for col in sector_columns)
print(f"\nTotal sector assignments: {total_assignments}")
print(f"Average per case: {total_assignments/total_cases:.2f}")

# If this were random with equal probability, we'd expect:
# Each sector has probability p of being assigned
# For 15 sectors to each have exactly 500 out of 1543 cases
# This would be p = 500/1543 = 0.324

expected_p = 500 / total_cases
print(f"\nImplied probability of each sector (except informal): {expected_p:.3f} (32.4%)")
print(f"Implied probability of informal: {165/total_cases:.3f} ({165/total_cases*100:.1f}%)")

# Check if assignment looks random or systematic
print("\n" + "=" * 80)
print("TESTING FOR SYNTHETIC PATTERNS")
print("=" * 80)

# Look at cases with many sectors
print("\nCases with 12+ sectors (suspicious high-assignment cases):")
high_sector_cases = df[df[sector_columns].sum(axis=1) >= 12]
print(f"Count: {len(high_sector_cases)}")

# Show which sectors are MISSING in these high-assignment cases
print("\nWhich sectors are excluded in high-assignment cases?")
for idx, row in high_sector_cases.head(20).iterrows():
    missing = [col for col in sector_columns if row[col] == 0]
    num_sectors = sum(row[col] for col in sector_columns)
    print(f"  {num_sectors} sectors, missing: {missing}")

# Check if "informal" is systematically excluded
print("\n" + "=" * 80)
print("IS 'INFORMAL' SYSTEMATICALLY EXCLUDED?")
print("=" * 80)

cases_with_informal = df[df['informal'] == 1]
cases_without_informal = df[df['informal'] == 0]

print(f"\nCases with informal: {len(cases_with_informal)}")
print(f"Cases without informal: {len(cases_without_informal)}")

print(f"\nAverage sectors in cases WITH informal: {cases_with_informal[sector_columns].sum(axis=1).mean():.2f}")
print(f"Average sectors in cases WITHOUT informal: {cases_without_informal[sector_columns].sum(axis=1).mean():.2f}")

# Check the sector pattern in cases with exactly 15 sectors
cases_15_sectors = df[df[sector_columns].sum(axis=1) == 15]
print(f"\n\nCases with exactly 15 sectors: {len(cases_15_sectors)}")
print("These cases should all have informal=0. Let's verify:")
print(f"Informal=0 in all 15-sector cases: {(cases_15_sectors['informal'] == 0).all()}")

# Check if there's a year pattern
print("\n" + "=" * 80)
print("TEMPORAL PATTERN")
print("=" * 80)
print("\nAre recent years more likely to have artificial balancing?")
year_stats = df.groupby('Case Filing Year for Action').agg({
    'agriculture': 'sum',
    'informal': 'sum',
    'Case URL': 'count'
}).rename(columns={'Case URL': 'total_cases'})

# Calculate average sectors per case by year
df['total_sectors'] = df[sector_columns].sum(axis=1)
avg_by_year = df.groupby('Case Filing Year for Action')['total_sectors'].mean()
year_stats['avg_sectors_per_case'] = avg_by_year

print(year_stats.tail(10))

# Final check: What if we remove all cases with 12+ sectors?
print("\n" + "=" * 80)
print("WHAT IF WE ONLY USE CASES WITH < 12 SECTORS?")
print("=" * 80)

clean_df = df[df[sector_columns].sum(axis=1) < 12]
print(f"\nCases with <12 sectors: {len(clean_df)} ({len(clean_df)/len(df)*100:.1f}%)")

print("\nSector counts in 'cleaner' subset:")
for col in sector_columns:
    count = clean_df[col].sum()
    print(f"  {col:30s}: {int(count)}")
