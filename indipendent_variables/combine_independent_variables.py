import pandas as pd
import numpy as np

# Define paths
base_path = "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space/indipendent_variables/"

# 1. Load Literacy Rate
print("Loading Literacy Rate...")
literacy = pd.read_csv(base_path + "Literacy Rate.csv")
print(f"  Shape: {literacy.shape}")
print(f"  Columns: {literacy.columns.tolist()}")

# 2. Load Corruption Index
print("\nLoading Corruption Index...")
corruption = pd.read_csv(base_path + "percived_corruption_index.csv")
print(f"  Shape: {corruption.shape}")
print(f"  Columns: {corruption.columns.tolist()}")

# 3. Load Press Freedom Index
print("\nLoading Press Freedom Index...")
press_freedom = pd.read_csv(base_path + "Press Freedom Index.csv", encoding='latin-1')
print(f"  Shape: {press_freedom.shape}")
print(f"  Columns: {press_freedom.columns.tolist()}")

# 4. Load Rule of Law
print("\nLoading Rule of Law...")
rule_of_law_wide = pd.read_csv(base_path + "wjp_rule_of_law_cleaned.csv")
print(f"  Shape: {rule_of_law_wide.shape}")

# ===== PROCESS LITERACY RATE =====
# Keep only country and literacy rate, rename columns
literacy_clean = literacy[['Country', 'Literacy Rate']].copy()
literacy_clean.columns = ['Country', 'Literacy_Rate']
# Convert to percentage (0-100 scale) for consistency
literacy_clean['Literacy_Rate_Pct'] = literacy_clean['Literacy_Rate'] * 100

# ===== PROCESS CORRUPTION INDEX =====
# Keep only ISO3, Country, and Score (drop Rank)
corruption_clean = corruption[['Country / Territory', 'ISO3', 'CPI 2024 score']].copy()
corruption_clean.columns = ['Country', 'ISO3', 'Corruption_Score']

# ===== PROCESS PRESS FREEDOM INDEX =====
# Filter for most recent year and keep score (drop rank)
latest_year = press_freedom['Year'].max()
print(f"\nUsing Press Freedom data from year: {latest_year}")
press_clean = press_freedom[press_freedom['Year'] == latest_year][['Country', 'ISO', 'Score']].copy()
press_clean.columns = ['Country', 'ISO3', 'Press_Freedom_Score']

# ===== PROCESS RULE OF LAW =====
# Transpose the data so countries become rows
# First, get the overall index score (row with "Factor 1: Limited Government Powers")
# Actually, let's use the overall WJP ROL Index score if available, or aggregate the factors

# Find the rows with the main factors
factor_rows = rule_of_law_wide[rule_of_law_wide['Country'].str.contains('Factor', na=False)]
print(f"\nFound {len(factor_rows)} factor rows in Rule of Law data")

# For simplicity, let's use Factor 1 (Limited Government Powers) as a representative
# Get row index where Country == "Factor 1: Limited Government Powers"
factor1_idx = rule_of_law_wide[rule_of_law_wide['Country'] == 'Factor 1: Limited Government Powers'].index[0]
factor1_data = rule_of_law_wide.iloc[factor1_idx, 2:].copy()  # Skip 'sheet_name' and 'Country' columns

# Create a dataframe with country names as rows
rule_of_law_clean = pd.DataFrame({
    'Country': factor1_data.index,
    'Rule_of_Law_Score': factor1_data.values
})

# Convert score to numeric (handle any non-numeric values)
rule_of_law_clean['Rule_of_Law_Score'] = pd.to_numeric(rule_of_law_clean['Rule_of_Law_Score'], errors='coerce')
# Convert from 0-1 scale to 0-100 scale for consistency
rule_of_law_clean['Rule_of_Law_Score_Pct'] = rule_of_law_clean['Rule_of_Law_Score'] * 100

print(f"\nRule of Law processed: {len(rule_of_law_clean)} countries")

# ===== ADD ISO3 CODES TO MISSING DATASETS =====
print("\n=== Adding ISO3 codes ===")

# Use corruption and press freedom data as reference for ISO3 codes
iso3_mapping = pd.concat([
    corruption_clean[['Country', 'ISO3']],
    press_clean[['Country', 'ISO3']]
]).drop_duplicates(subset='Country', keep='first')

# Add ISO3 to literacy data
literacy_clean = literacy_clean.merge(iso3_mapping, on='Country', how='left')

# Add ISO3 to rule of law data
rule_of_law_clean = rule_of_law_clean.merge(iso3_mapping, on='Country', how='left')

# Check for countries without ISO3 codes
print(f"\nLiteracy countries without ISO3: {literacy_clean['ISO3'].isna().sum()}")
print(f"Rule of Law countries without ISO3: {rule_of_law_clean['ISO3'].isna().sum()}")

# ===== COMBINE ALL DATASETS =====
print("\n=== Combining all datasets ===")

# Start with corruption data as the base (it has the most complete ISO3 codes)
combined = corruption_clean[['ISO3', 'Country', 'Corruption_Score']].copy()

# Merge press freedom
combined = combined.merge(
    press_clean[['ISO3', 'Press_Freedom_Score']],
    on='ISO3',
    how='outer'
)

# Merge literacy
combined = combined.merge(
    literacy_clean[['ISO3', 'Literacy_Rate_Pct']],
    on='ISO3',
    how='outer'
)

# Merge rule of law
combined = combined.merge(
    rule_of_law_clean[['ISO3', 'Rule_of_Law_Score_Pct']],
    on='ISO3',
    how='outer'
)

# Remove rows where ISO3 is missing
combined = combined.dropna(subset=['ISO3'])

# Reorder columns: ISO3 first, then Country, then all scores
combined = combined[['ISO3', 'Country', 'Corruption_Score', 'Press_Freedom_Score',
                     'Literacy_Rate_Pct', 'Rule_of_Law_Score_Pct']]

# Sort by ISO3
combined = combined.sort_values('ISO3').reset_index(drop=True)

print(f"\nFinal combined dataset shape: {combined.shape}")
print(f"Countries with complete data (all 4 variables): {combined.dropna().shape[0]}")
print(f"\nMissing values per variable:")
print(combined.isnull().sum())

# ===== SAVE TO CSV =====
output_path = "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space/Workfolder Climate cases (Caspar)/combined_independent_variables.csv"
combined.to_csv(output_path, index=False)
print(f"\n✓ Combined data saved to: {output_path}")

# Display first few rows
print("\nFirst 10 rows of combined data:")
print(combined.head(10).to_string())

print("\n" + "="*60)
print("NOTES ABOUT THE DATA:")
print("="*60)
print("1. RANKINGS DROPPED: Only kept actual scores/values, not rankings")
print("2. NORMALIZATION: ")
print("   - Corruption Score: 0-100 (higher = less corrupt)")
print("   - Press Freedom Score: 0-100 (higher = more freedom)")
print("   - Literacy Rate: 0-100 (percentage)")
print("   - Rule of Law Score: 0-100 (converted from 0-1 scale)")
print("\n3. All scores are now comparable percentages (0-100 scale)")
print("4. Higher values = better performance for all variables")
