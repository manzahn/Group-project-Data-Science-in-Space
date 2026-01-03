#!/usr/bin/env python3
"""
Create a simplified Land Matrix dataset similar to ISDS Lawsuit structure
"""

import pandas as pd
import os

# Define file paths
base_path = "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space/legacy land matrix"
deals_locations_file = os.path.join(base_path, "deals_locations_merged.csv")
output_file = os.path.join(base_path, "land_matrix_simplified.csv")

print("Loading merged deals+locations data...")
df = pd.read_csv(deals_locations_file, delimiter=';', low_memory=False)
print(f"Loaded {len(df)} rows")

# Select and rename columns to match ISDS-style structure
simplified_df = pd.DataFrame({
    'Deal ID': df['Deal ID'],
    'Respondent State': df['Target country'],
    'Home State of Investor': df['Operating company: Investor Country'],
    'Economic Sector': df['Intention of investment'],
    'Negotiation Status': df['Current negotiation status'],
    'Implementation Status': df['Current implementation status'],
    'Purchase Price': df['Purchase price'],
    'Purchase Price Currency': df['Purchase price currency'],
    'Annual Leasing Fee': df['Annual leasing fee'],
    'Annual Leasing Fee Currency': df['Annual leasing fee currency'],
    'Deal Size (ha)': df['Deal size'],
    'Contract Year': df['Created at'],  # Using created date as year
    'Location': df['Location'],
    'Spatial Accuracy': df['Spatial accuracy level'],
    'Point Coordinates': df['Point'],
    'Facility Name': df['Facility name'],
    'Operating Company': df['Operating company: Name'],
    'Deal Scope': df['Deal scope'],
})

# Save simplified dataset
print(f"\nSaving simplified dataset with {len(simplified_df)} rows and {len(simplified_df.columns)} columns...")
simplified_df.to_csv(output_file, index=False, sep=';')
print(f"Saved: {output_file}")

# Print summary statistics
print("\n" + "="*80)
print("SIMPLIFIED DATASET SUMMARY")
print("="*80)

print(f"\nTotal rows: {len(simplified_df)}")
print(f"Total columns: {len(simplified_df.columns)}")

print("\n--- Column List ---")
for i, col in enumerate(simplified_df.columns, 1):
    print(f"{i}. {col}")

print("\n--- Sample Data (first 5 rows) ---")
print(simplified_df.head().to_string())

print("\n--- Negotiation Status Distribution ---")
print(simplified_df['Negotiation Status'].value_counts())

print("\n--- Implementation Status Distribution ---")
print(simplified_df['Implementation Status'].value_counts())

print("\n--- Top 10 Respondent States ---")
print(simplified_df['Respondent State'].value_counts().head(10))

print("\n--- Top 10 Home States of Investors ---")
print(simplified_df['Home State of Investor'].value_counts().head(10))

print("\n--- Top 10 Economic Sectors ---")
print(simplified_df['Economic Sector'].value_counts().head(10))

print("\n" + "="*80)
print("Done!")
print("="*80)
