import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("Loading data and preparing geographic boundaries...")

# =====================================================================
# 1. LOAD AND AGGREGATE LOAN DATA
# =====================================================================
filepath = "C:\\Users\\Adams\\Downloads\\Model\\loan.csv"
df = pd.read_csv(filepath, low_memory=False, usecols=['loan_status', 'addr_state'])

# Define bad loan statuses
bad_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
df = df[~df['loan_status'].isin(['Current', 'In Grace Period', 'Late (16-30 days)'])]
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)

# Calculate default rate by state
state_risk = df.groupby('addr_state')['loan_status'].agg(
	total_loans='count',
	default_rate=lambda x: x.mean() * 100
).reset_index()

# Filter low-volume states to eliminate noise
state_risk = state_risk[state_risk['total_loans'] > 100]

# =====================================================================
# 2. MATCH WITH GEOGRAPHIC SHAPES
# =====================================================================
# Pulls a standard, lightweight US States map dataset directly via URL
usa_geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/us-states.json"
states_gdf = gpd.read_file(usa_geojson_url)

# Merge the geographic shapes with your loan data
# 'id' in the GeoJSON matches 'addr_state' (e.g., 'CA', 'NY', 'TX')
merged = states_gdf.merge(state_risk, left_on='id', right_on='addr_state', how='inner')

# Filter out Alaska (AK) and Hawaii (HI) to keep the continental US view clean and compact
merged = merged[~merged['id'].isin(['AK', 'HI'])]

# =====================================================================
# 3. GENERATE NATIVE POP-UP WINDOW
# =====================================================================
print("Opening map window...")

# Create the plot with a dark theme background
fig, ax = plt.subplots(1, 1, figsize=(12, 7), facecolor='#1e1e1e')
ax.set_facecolor('#1e1e1e')

# Render the choropleth map
merged.plot(
	column='default_rate', 
	cmap='Blues', 
	linewidth=0.6, 
	ax=ax, 
	edgecolor='#2c2c2c', 
	legend=True,
	legend_kwds={'label': "Default Rate (%)", 'orientation': "horizontal", 'pad': 0.05}
)

# Style details
ax.axis('off')
plt.title("Historical Loan Default Rate by State (%)", color='white', fontsize=14, pad=10)

# Adjust the color bar text to match the dark theme
cax = fig.get_axes()[1]
cax.xaxis.label.set_color('white')
cax.tick_params(colors='white')

# This forces the native desktop window to pop up directly
plt.show()
print("Process finished.")
