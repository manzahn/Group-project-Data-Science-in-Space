import pandas as pd

# Read the original CSV
df = pd.read_csv('combined_climate_data.csv')

sector_columns = [
    'agriculture', 'construction', 'education', 'energy_supply',
    'enviromental_management', 'extractive', 'finance_insurance',
    'fishing', 'forestry', 'informal', 'manufacturing', 'media',
    'public', 'trade', 'transportation_logistics', 'water_management'
]

df['num_sectors'] = df[sector_columns].sum(axis=1)

print("=" * 80)
print("CREATING CLEANED DATASETS")
print("=" * 80)
print(f"\nOriginal dataset: {len(df)} cases")

# Strategy 1: Remove cases with 12+ sectors (removes artificial inflation)
print("\n" + "=" * 80)
print("STRATEGY 1: Cases with <12 sectors (Conservative approach)")
print("=" * 80)
df_clean_sectors = df[df['num_sectors'] < 12].copy()
print(f"Cases remaining: {len(df_clean_sectors)} ({len(df_clean_sectors)/len(df)*100:.1f}%)")
print(f"Average sectors per case: {df_clean_sectors['num_sectors'].mean():.2f}")

print("\nSector distribution:")
for col in sector_columns:
    count = df_clean_sectors[col].sum()
    print(f"  {col:30s}: {int(count)}")

# Save this version
df_clean_sectors.to_csv('combined_climate_data_CLEANED_lt12sectors.csv', index=False)
print("\n✓ Saved: combined_climate_data_CLEANED_lt12sectors.csv")

# Strategy 2: Only pre-2022 cases (before artificial balancing started)
print("\n" + "=" * 80)
print("STRATEGY 2: Cases filed before 2022 (Temporal filter)")
print("=" * 80)
df_clean_temporal = df[df['Case Filing Year for Action'] < 2022].copy()
print(f"Cases remaining: {len(df_clean_temporal)} ({len(df_clean_temporal)/len(df)*100:.1f}%)")
print(f"Average sectors per case: {df_clean_temporal['num_sectors'].mean():.2f}")

print("\nSector distribution:")
for col in sector_columns:
    count = df_clean_temporal[col].sum()
    print(f"  {col:30s}: {int(count)}")

# Save this version
df_clean_temporal.to_csv('combined_climate_data_CLEANED_pre2022.csv', index=False)
print("\n✓ Saved: combined_climate_data_CLEANED_pre2022.csv")

# Strategy 3: Combination - pre-2022 OR single-sector cases from any year
print("\n" + "=" * 80)
print("STRATEGY 3: Pre-2022 OR single-sector (Balanced approach)")
print("=" * 80)
df_clean_combo = df[
    (df['Case Filing Year for Action'] < 2022) |
    (df['num_sectors'] == 1)
].copy()
print(f"Cases remaining: {len(df_clean_combo)} ({len(df_clean_combo)/len(df)*100:.1f}%)")
print(f"Average sectors per case: {df_clean_combo['num_sectors'].mean():.2f}")

print("\nSector distribution:")
for col in sector_columns:
    count = df_clean_combo[col].sum()
    print(f"  {col:30s}: {int(count)}")

# Save this version
df_clean_combo.to_csv('combined_climate_data_CLEANED_pre2022_or_single.csv', index=False)
print("\n✓ Saved: combined_climate_data_CLEANED_pre2022_or_single.csv")

# Strategy 4: RECOMMENDED - Cases with <=6 sectors (removes extreme inflation)
print("\n" + "=" * 80)
print("STRATEGY 4: Cases with ≤6 sectors (RECOMMENDED)")
print("=" * 80)
df_clean_recommended = df[df['num_sectors'] <= 6].copy()
print(f"Cases remaining: {len(df_clean_recommended)} ({len(df_clean_recommended)/len(df)*100:.1f}%)")
print(f"Average sectors per case: {df_clean_recommended['num_sectors'].mean():.2f}")

print("\nSector distribution:")
for col in sector_columns:
    count = df_clean_recommended[col].sum()
    print(f"  {col:30s}: {int(count)}")

print("\nDistribution of sectors per case:")
sector_dist = df_clean_recommended['num_sectors'].value_counts().sort_index()
for num, count in sector_dist.items():
    print(f"  {int(num)} sectors: {count} cases")

# Save this version
df_clean_recommended.to_csv('combined_climate_data_CLEANED_RECOMMENDED.csv', index=False)
print("\n✓ Saved: combined_climate_data_CLEANED_RECOMMENDED.csv")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"\n{'Dataset':<50} {'Cases':<10} {'Avg Sectors':<15} {'Coverage'}")
print("-" * 90)
print(f"{'Original':<50} {len(df):<10} {df['num_sectors'].mean():<15.2f} 100.0%")
print(f"{'<12 sectors (Strategy 1)':<50} {len(df_clean_sectors):<10} {df_clean_sectors['num_sectors'].mean():<15.2f} {len(df_clean_sectors)/len(df)*100:.1f}%")
print(f"{'Pre-2022 (Strategy 2)':<50} {len(df_clean_temporal):<10} {df_clean_temporal['num_sectors'].mean():<15.2f} {len(df_clean_temporal)/len(df)*100:.1f}%")
print(f"{'Pre-2022 OR single-sector (Strategy 3)':<50} {len(df_clean_combo):<10} {df_clean_combo['num_sectors'].mean():<15.2f} {len(df_clean_combo)/len(df)*100:.1f}%")
print(f"{'≤6 sectors - RECOMMENDED (Strategy 4)':<50} {len(df_clean_recommended):<10} {df_clean_recommended['num_sectors'].mean():<15.2f} {len(df_clean_recommended)/len(df)*100:.1f}%")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
Use: combined_climate_data_CLEANED_RECOMMENDED.csv

This dataset:
- Retains 68% of cases (1062 cases)
- Removes extreme multi-sector inflation (12+ sectors)
- Keeps realistic 2-6 sector combinations
- Average 2.75 sectors/case (much more realistic than 4.97)
- No artificial "500" balancing
""")
