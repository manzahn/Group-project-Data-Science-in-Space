import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_venn import venn2, venn3
from itertools import combinations
import seaborn as sns

# Read the cleaned recommended dataset
df = pd.read_csv('combined_climate_data_CLEANED_RECOMMENDED.csv')

sector_columns = [
    'agriculture', 'construction', 'education', 'energy_supply',
    'enviromental_management', 'extractive', 'finance_insurance',
    'fishing', 'forestry', 'informal', 'manufacturing', 'media',
    'public', 'trade', 'transportation_logistics', 'water_management'
]

print("=" * 80)
print("VENN-STYLE ANALYSIS ON CLEANED DATASET")
print("=" * 80)
print(f"Total cases: {len(df)}")

# Get top 5 sectors by count
sector_counts = {}
for col in sector_columns:
    sector_counts[col] = df[col].sum()

top_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 sectors:")
for sector, count in top_sectors:
    print(f"  {sector}: {count}")

top_sector_names = [s[0] for s in top_sectors]

# VISUALIZATION 1: Traditional Venn for Top 3 Sectors
print("\n" + "=" * 80)
print("CREATING VENN DIAGRAM - Top 3 Sectors")
print("=" * 80)

top_3 = top_sector_names[:3]
print(f"Analyzing: {', '.join(top_3)}")

# Calculate overlaps
only_A = len(df[(df[top_3[0]] == 1) & (df[top_3[1]] == 0) & (df[top_3[2]] == 0)])
only_B = len(df[(df[top_3[0]] == 0) & (df[top_3[1]] == 1) & (df[top_3[2]] == 0)])
only_C = len(df[(df[top_3[0]] == 0) & (df[top_3[1]] == 0) & (df[top_3[2]] == 1)])
AB_not_C = len(df[(df[top_3[0]] == 1) & (df[top_3[1]] == 1) & (df[top_3[2]] == 0)])
AC_not_B = len(df[(df[top_3[0]] == 1) & (df[top_3[1]] == 0) & (df[top_3[2]] == 1)])
BC_not_A = len(df[(df[top_3[0]] == 0) & (df[top_3[1]] == 1) & (df[top_3[2]] == 1)])
ABC = len(df[(df[top_3[0]] == 1) & (df[top_3[1]] == 1) & (df[top_3[2]] == 1)])

print(f"\nOnly {top_3[0]}: {only_A}")
print(f"Only {top_3[1]}: {only_B}")
print(f"Only {top_3[2]}: {only_C}")
print(f"{top_3[0]} & {top_3[1]}: {AB_not_C}")
print(f"{top_3[0]} & {top_3[2]}: {AC_not_B}")
print(f"{top_3[1]} & {top_3[2]}: {BC_not_A}")
print(f"All three: {ABC}")

fig, ax = plt.subplots(figsize=(12, 8))
venn = venn3(subsets=(only_A, only_B, AB_not_C, only_C, AC_not_B, BC_not_A, ABC),
             set_labels=(top_3[0].replace('_', ' ').title(),
                        top_3[1].replace('_', ' ').title(),
                        top_3[2].replace('_', ' ').title()),
             ax=ax)

# Customize colors
if venn.get_patch_by_id('100'):
    venn.get_patch_by_id('100').set_color('#1f77b4')
    venn.get_patch_by_id('100').set_alpha(0.5)
if venn.get_patch_by_id('010'):
    venn.get_patch_by_id('010').set_color('#ff7f0e')
    venn.get_patch_by_id('010').set_alpha(0.5)
if venn.get_patch_by_id('001'):
    venn.get_patch_by_id('001').set_color('#2ca02c')
    venn.get_patch_by_id('001').set_alpha(0.5)

plt.title(f'Sector Overlap: Top 3 Sectors\nCleaned Dataset (n={len(df)} cases)',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('venn_top3_sectors.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: venn_top3_sectors.png")
plt.close()

# VISUALIZATION 2: Co-occurrence Matrix Heatmap
print("\n" + "=" * 80)
print("CREATING CO-OCCURRENCE HEATMAP")
print("=" * 80)

# Calculate co-occurrence matrix
cooccurrence = np.zeros((len(sector_columns), len(sector_columns)))

for i, sector_i in enumerate(sector_columns):
    for j, sector_j in enumerate(sector_columns):
        if i == j:
            # Diagonal: total count
            cooccurrence[i][j] = df[sector_i].sum()
        else:
            # Co-occurrence count
            cooccurrence[i][j] = len(df[(df[sector_i] == 1) & (df[sector_j] == 1)])

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 12))

# Clean sector names for display
clean_names = [s.replace('_', ' ').title() for s in sector_columns]

sns.heatmap(cooccurrence,
            xticklabels=clean_names,
            yticklabels=clean_names,
            annot=True,
            fmt='.0f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Co-occurrence Count'},
            ax=ax,
            square=True)

plt.title('Sector Co-occurrence Matrix\nCleaned Dataset (Diagonal = Total Count, Off-diagonal = Co-occurrences)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('')
plt.ylabel('')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('cooccurrence_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cooccurrence_heatmap.png")
plt.close()

# VISUALIZATION 3: Chord-style diagram data (for manual plotting or other tools)
print("\n" + "=" * 80)
print("CREATING SECTOR NETWORK DATA")
print("=" * 80)

# Get top co-occurrences
cooccurrence_pairs = []
for i in range(len(sector_columns)):
    for j in range(i+1, len(sector_columns)):
        count = int(cooccurrence[i][j])
        if count > 0:
            cooccurrence_pairs.append({
                'sector1': sector_columns[i],
                'sector2': sector_columns[j],
                'count': count
            })

cooccurrence_df = pd.DataFrame(cooccurrence_pairs).sort_values('count', ascending=False)
print("\nTop 20 sector pairs:")
print(cooccurrence_df.head(20).to_string(index=False))

# Save to CSV for reference
cooccurrence_df.to_csv('sector_cooccurrences.csv', index=False)
print("\n✓ Saved: sector_cooccurrences.csv")

# VISUALIZATION 4: UpSet-style representation
print("\n" + "=" * 80)
print("CREATING UPSET-STYLE VISUALIZATION")
print("=" * 80)

# Get all unique combinations
from collections import Counter

combinations_list = []
for idx, row in df.iterrows():
    active = tuple(sorted([col for col in sector_columns if row[col] == 1]))
    if active:
        combinations_list.append(active)

combo_counts = Counter(combinations_list)
top_combos = combo_counts.most_common(15)

print("\nTop 15 sector combinations:")
for i, (combo, count) in enumerate(top_combos, 1):
    if len(combo) == 0:
        combo_str = "None"
    else:
        combo_str = " + ".join([s.replace('_', ' ').title() for s in combo])
    print(f"{i:2d}. ({count:3d} cases) {combo_str}")

# Create UpSet-style bar chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                gridspec_kw={'height_ratios': [3, 1]})

# Top: Bar chart of combination sizes
combo_labels = []
combo_sizes = []
combo_counts_list = []

for combo, count in top_combos:
    if len(combo) == 0:
        combo_labels.append("None")
    elif len(combo) == 1:
        combo_labels.append(combo[0].replace('_', '\n'))
    else:
        combo_labels.append(f"{len(combo)}\nsectors")
    combo_sizes.append(len(combo))
    combo_counts_list.append(count)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_combos)))
ax1.bar(range(len(top_combos)), combo_counts_list, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Number of Cases', fontsize=12, fontweight='bold')
ax1.set_title('Top 15 Sector Combinations (UpSet Style)\nCleaned Dataset',
              fontsize=14, fontweight='bold', pad=20)
ax1.set_xticks(range(len(top_combos)))
ax1.set_xticklabels(combo_labels, fontsize=9, rotation=0)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(combo_counts_list):
    ax1.text(i, v + max(combo_counts_list)*0.02, str(v),
             ha='center', va='bottom', fontweight='bold', fontsize=10)

# Bottom: Show number of sectors in each combination
ax2.bar(range(len(top_combos)), combo_sizes, color='steelblue', edgecolor='black', linewidth=1.5)
ax2.set_ylabel('# Sectors', fontsize=12, fontweight='bold')
ax2.set_xlabel('Combination Rank', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(top_combos)))
ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_combos))], fontsize=9)
ax2.set_ylim(0, max(combo_sizes) + 1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('upset_style_combinations.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: upset_style_combinations.png")
plt.close()

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nGenerated visualizations:")
print("  1. venn_top3_sectors.png - Traditional Venn diagram for top 3 sectors")
print("  2. cooccurrence_heatmap.png - Full co-occurrence matrix")
print("  3. upset_style_combinations.png - Top 15 sector combinations")
print("  4. sector_cooccurrences.csv - Pairwise co-occurrence data")
