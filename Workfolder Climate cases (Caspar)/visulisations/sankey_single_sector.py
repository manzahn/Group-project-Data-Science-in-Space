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

# Filter to single-sector cases only
single_sector_df = df[df['num_sectors'] == 1].copy()

# Identify which sector each case belongs to
def get_sector(row):
    for col in sector_columns:
        if row[col] == 1:
            return col
    return None

single_sector_df['sector'] = single_sector_df.apply(get_sector, axis=1)

print(f"Analyzing {len(single_sector_df)} single-sector cases ({len(single_sector_df)/len(df)*100:.1f}% of total dataset)")
print("=" * 80)

# Prepare Sankey data
labels = []
sources = []
targets = []
values = []
colors_links = []

# Node 0: All Single-Sector Cases
labels.append(f'Single-Sector Cases\n({len(single_sector_df)})')
current_node = 1

# First level: By Sector
sector_counts = single_sector_df['sector'].value_counts()
sector_nodes = {}

print("\nSector Distribution:")
for sector in sector_counts.index:
    count = sector_counts[sector]
    sector_name = sector.replace('_', ' ').title()
    labels.append(f'{sector_name}\n({count})')
    sector_nodes[sector] = current_node

    sources.append(0)
    targets.append(current_node)
    values.append(count)
    colors_links.append('rgba(46, 134, 171, 0.4)')

    print(f"  {sector_name}: {count} cases")
    current_node += 1

# Second level: By Status for each major sector
# Focus on top 5 sectors
top_sectors = sector_counts.head(5).index

status_nodes = {}
print("\nStatus Distribution by Top Sectors:")

for sector in top_sectors:
    sector_data = single_sector_df[single_sector_df['sector'] == sector]
    status_counts = sector_data['Status'].value_counts()

    sector_name = sector.replace('_', ' ').title()
    print(f"\n{sector_name}:")

    for status, count in status_counts.items():
        # Create unique label for status under this sector
        status_label = f'{status}\n({count})'

        # Check if we already have this status label
        if status_label not in labels:
            labels.append(status_label)
            status_nodes[f'{sector}_{status}'] = current_node
        else:
            # If duplicate label, make it unique
            status_label = f'{status} ({sector_name[:3]})\n({count})'
            labels.append(status_label)
            status_nodes[f'{sector}_{status}'] = current_node

        sources.append(sector_nodes[sector])
        targets.append(current_node)
        values.append(count)

        # Color by status
        if 'Decided' in status:
            color = 'rgba(76, 175, 80, 0.4)'  # Green
        elif 'Pending' in status:
            color = 'rgba(255, 152, 0, 0.4)'  # Orange
        elif 'Dismissed' in status or 'Withdrawn' in status:
            color = 'rgba(244, 67, 54, 0.4)'  # Red
        else:
            color = 'rgba(158, 158, 158, 0.4)'  # Gray

        colors_links.append(color)

        print(f"  {status}: {count}")
        current_node += 1

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,
        thickness=25,
        line=dict(color="black", width=0.5),
        label=labels,
        color=['#2E86AB'] + ['#6CA6C1'] * len(sector_counts) + ['#A8DADC'] * (len(labels) - len(sector_counts) - 1)
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
        'text': f"Single-Sector Climate Cases Analysis ({len(single_sector_df)} cases, {len(single_sector_df)/len(df)*100:.1f}% of dataset)<br>" +
                "<sub>Focusing on cases with clear, single-sector attribution for accuracy</sub>",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20}
    },
    font=dict(size=11),
    height=900,
    width=1600
)

# Save the figure
fig.write_html('sankey_single_sector_only.html')
print("\n" + "=" * 80)
print("Sankey diagram saved to 'sankey_single_sector_only.html'")

# Show the figure
fig.show()

# Additional statistics
print("\n" + "=" * 80)
print("ADDITIONAL INSIGHTS")
print("=" * 80)

# Overall status distribution
print("\nOverall Status Distribution (Single-Sector Cases):")
status_dist = single_sector_df['Status'].value_counts()
for status, count in status_dist.items():
    print(f"  {status}: {count} ({count/len(single_sector_df)*100:.1f}%)")

# Year distribution for top sectors
print("\nFiling Year Trends (Top 3 Sectors):")
for sector in top_sectors[:3]:
    sector_data = single_sector_df[single_sector_df['sector'] == sector]
    sector_name = sector.replace('_', ' ').title()
    print(f"\n{sector_name}:")

    year_counts = sector_data['Case Filing Year for Action'].value_counts().sort_index(ascending=False).head(5)
    for year, count in year_counts.items():
        print(f"  {year}: {count} cases")

# Geography insights
print("\nTop Geographies (Single-Sector Cases):")
geo_counts = single_sector_df['Geographies'].value_counts().head(10)
for geo, count in geo_counts.items():
    print(f"  {geo}: {count} cases")
