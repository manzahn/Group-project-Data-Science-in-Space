#!/usr/bin/env python3
"""
Script to merge Land Matrix CSV files.
Creates three merged tables:
1. deals_locations.csv - Deals with location information
2. deals_contracts.csv - Deals with contract information
3. deals_datasources.csv - Deals with data source information

All tables include investor details merged from investors.csv
"""

import pandas as pd
import os

# Define file paths
base_path = "/home/caspar/Coding workspaces/Spacial DATASCIENCE/Group-project-Data-Science-in-Space/legacy land matrix"

deals_file = os.path.join(base_path, "deals.csv")
locations_file = os.path.join(base_path, "locations.csv")
contracts_file = os.path.join(base_path, "contracts.csv")
datasources_file = os.path.join(base_path, "datasources.csv")
investors_file = os.path.join(base_path, "investors.csv")
involvements_file = os.path.join(base_path, "involvements.csv")

print("Loading CSV files...")

# Load all CSV files with semicolon delimiter
deals = pd.read_csv(deals_file, delimiter=';', low_memory=False)
locations = pd.read_csv(locations_file, delimiter=';', low_memory=False)
contracts = pd.read_csv(contracts_file, delimiter=';', low_memory=False)
datasources = pd.read_csv(datasources_file, delimiter=';', low_memory=False)
investors = pd.read_csv(investors_file, delimiter=';', low_memory=False)

print(f"Loaded {len(deals)} deals")
print(f"Loaded {len(locations)} locations")
print(f"Loaded {len(contracts)} contracts")
print(f"Loaded {len(datasources)} data sources")
print(f"Loaded {len(investors)} investors")

# Add prefix to investor columns to avoid conflicts
investors_renamed = investors.rename(columns={
    'Name': 'Operating company: Investor Name',
    'Country of registration/origin': 'Operating company: Investor Country',
    'Classification': 'Operating company: Investor Classification',
    'Investor homepage': 'Operating company: Investor Homepage',
    'Opencorporates link': 'Operating company: Investor Opencorporates',
    'Comment': 'Operating company: Investor Comment',
    'Action comment': 'Operating company: Investor Action Comment'
})

# Merge deals with investor information
print("\nMerging deals with investor information...")
deals_with_investors = deals.merge(
    investors_renamed,
    left_on='Operating company: Investor ID',
    right_on='Investor ID',
    how='left',
    suffixes=('', '_investor')
)

print(f"Deals with investor info: {len(deals_with_investors)} rows")

# Create merged tables
print("\nCreating merged tables...")

# 1. Deals + Locations
deals_locations = deals_with_investors.merge(
    locations,
    on='Deal ID',
    how='left',
    suffixes=('', '_location')
)
print(f"Deals + Locations: {len(deals_locations)} rows, {len(deals_locations.columns)} columns")

# 2. Deals + Contracts
deals_contracts = deals_with_investors.merge(
    contracts,
    on='Deal ID',
    how='left',
    suffixes=('', '_contract')
)
print(f"Deals + Contracts: {len(deals_contracts)} rows, {len(deals_contracts.columns)} columns")

# 3. Deals + Data Sources
deals_datasources = deals_with_investors.merge(
    datasources,
    on='Deal ID',
    how='left',
    suffixes=('', '_datasource')
)
print(f"Deals + Data Sources: {len(deals_datasources)} rows, {len(deals_datasources.columns)} columns")

# Save merged tables
print("\nSaving merged tables...")

output_path_locations = os.path.join(base_path, "deals_locations_merged.csv")
output_path_contracts = os.path.join(base_path, "deals_contracts_merged.csv")
output_path_datasources = os.path.join(base_path, "deals_datasources_merged.csv")

deals_locations.to_csv(output_path_locations, index=False, sep=';')
print(f"Saved: {output_path_locations}")

deals_contracts.to_csv(output_path_contracts, index=False, sep=';')
print(f"Saved: {output_path_contracts}")

deals_datasources.to_csv(output_path_datasources, index=False, sep=';')
print(f"Saved: {output_path_datasources}")

# Print column lists for each merged table
print("\n" + "="*80)
print("COLUMN LISTS FOR MERGED TABLES")
print("="*80)

print("\n1. DEALS_LOCATIONS_MERGED.CSV COLUMNS:")
print("-" * 80)
for i, col in enumerate(deals_locations.columns, 1):
    print(f"{i}. {col}")

print("\n2. DEALS_CONTRACTS_MERGED.CSV COLUMNS:")
print("-" * 80)
for i, col in enumerate(deals_contracts.columns, 1):
    print(f"{i}. {col}")

print("\n3. DEALS_DATASOURCES_MERGED.CSV COLUMNS:")
print("-" * 80)
for i, col in enumerate(deals_datasources.columns, 1):
    print(f"{i}. {col}")

print("\n" + "="*80)
print("Done! All merged files created successfully.")
print("="*80)
