import pandas as pd
import plotly.graph_objects as go
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

# Calculate number of sectors per case
df['num_sectors'] = df[sector_columns].sum(axis=1)

# Categorize cases
single_sector = df[df['num_sectors'] == 1]
multi_sector = df[df['num_sectors'] > 1]

print(f"Single-sector cases: {len(single_sector)}")
print(f"Multi-sector cases: {len(multi_sector)}")

# For single-sector cases, identify which sector
single_sector_breakdown = {}
for idx, row in single_sector.iterrows():
    for col in sector_columns:
        if row[col] == 1:
            single_sector_breakdown[col] = single_sector_breakdown.get(col, 0) + 1
            break

# For multi-sector cases, group by number of sectors
multi_sector_breakdown = multi_sector['num_sectors'].value_counts().sort_index().to_dict()

# Prepare Sankey diagram data
labels = ['All Cases']  # Node 0
sources = []
targets = []
values = []
colors_links = []

# Add main category nodes
labels.append('Single-sector cases (409)')  # Node 1
labels.append('Multi-sector cases (1,134)')  # Node 2

# Flow from All Cases to Single/Multi
sources.append(0)
targets.append(1)
values.append(len(single_sector))
colors_links.append('rgba(100, 150, 200, 0.4)')

sources.append(0)
targets.append(2)
values.append(len(multi_sector))
colors_links.append('rgba(200, 100, 150, 0.4)')

# Add single-sector breakdown (top sectors)
sorted_single = sorted(single_sector_breakdown.items(), key=lambda x: x[1], reverse=True)
node_idx = 3

# Top 8 single sectors + "Other"
top_single_sectors = sorted_single[:8]
other_single = sum([count for sector, count in sorted_single[8:]])

for sector, count in top_single_sectors:
    label = f"{sector.replace('_', ' ').title()} ({count})"
    labels.append(label)
    sources.append(1)  # From Single-sector cases
    targets.append(node_idx)
    values.append(count)
    colors_links.append('rgba(100, 150, 200, 0.3)')
    node_idx += 1

if other_single > 0:
    labels.append(f"Other single sectors ({other_single})")
    sources.append(1)
    targets.append(node_idx)
    values.append(other_single)
    colors_links.append('rgba(100, 150, 200, 0.2)')
    node_idx += 1

# Add multi-sector breakdown by number of sectors
# Group smaller counts
multi_breakdown_display = []
for num, count in sorted(multi_sector_breakdown.items()):
    if count >= 30:  # Show individually if >= 30 cases
        multi_breakdown_display.append((f"{num} sectors ({count})", count))
    else:
        # Add to appropriate group
        if num <= 3:
            # Find or create 2-3 sectors group
            found = False
            for i, (label, val) in enumerate(multi_breakdown_display):
                if label.startswith("2-3 sectors"):
                    multi_breakdown_display[i] = (f"2-3 sectors ({val + count})", val + count)
                    found = True
                    break
            if not found:
                multi_breakdown_display.append((f"2-3 sectors ({count})", count))
        elif num <= 6:
            found = False
            for i, (label, val) in enumerate(multi_breakdown_display):
                if label.startswith("4-6 sectors"):
                    multi_breakdown_display[i] = (f"4-6 sectors ({val + count})", val + count)
                    found = True
                    break
            if not found:
                multi_breakdown_display.append((f"4-6 sectors ({count})", count))
        elif num <= 11:
            found = False
            for i, (label, val) in enumerate(multi_breakdown_display):
                if label.startswith("7-11 sectors"):
                    multi_breakdown_display[i] = (f"7-11 sectors ({val + count})", val + count)
                    found = True
                    break
            if not found:
                multi_breakdown_display.append((f"7-11 sectors ({count})", count))
        else:
            found = False
            for i, (label, val) in enumerate(multi_breakdown_display):
                if label.startswith("12+ sectors"):
                    multi_breakdown_display[i] = (f"12+ sectors ({val + count})", val + count)
                    found = True
                    break
            if not found:
                multi_breakdown_display.append((f"12+ sectors ({count})", count))

# Sort and add to diagram
multi_breakdown_display = sorted(multi_breakdown_display, key=lambda x: x[1], reverse=True)
for label, count in multi_breakdown_display:
    labels.append(label)
    sources.append(2)  # From Multi-sector cases
    targets.append(node_idx)
    values.append(count)
    colors_links.append('rgba(200, 100, 150, 0.3)')
    node_idx += 1

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=['#2E86AB', '#A23B72', '#F18F01'] + ['#6CA6C1'] * 20 + ['#E07B91'] * 10
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=colors_links
    )
)])

fig.update_layout(
    title={
        'text': "Climate Cases: Single vs. Multi-Sector Distribution<br><sub>Most cases are multi-sector challenges affecting multiple economic areas</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    font=dict(size=12),
    height=800,
    width=1400
)

# Save the figure
fig.write_html('sankey_multisector.html')
print("\nSankey diagram saved to 'sankey_multisector.html'")

# Show the figure
fig.show()

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total cases: {len(df)}")
print(f"Single-sector: {len(single_sector)} ({len(single_sector)/len(df)*100:.1f}%)")
print(f"Multi-sector: {len(multi_sector)} ({len(multi_sector)/len(df)*100:.1f}%)")
print(f"\nTop single sectors:")
for sector, count in sorted_single[:5]:
    print(f"  {sector}: {count}")
print(f"\nMulti-sector distribution:")
for label, count in multi_breakdown_display:
    print(f"  {label}: {count}")
