import pandas as pd
from scipy.stats import chi2_contingency
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("Loading data... (Optimized to load only required columns)")
filepath = "C:\\Users\\Adams\\Downloads\\Model\\loan.csv"

# Load only the state and loan status columns to save memory and time
df = pd.read_csv(filepath, low_memory=False, usecols=['addr_state', 'loan_status'])

# =====================================================================
# 1. CLEAN TARGET VARIABLE
# =====================================================================
bad_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
df = df[~df['loan_status'].isin(['Current', 'In Grace Period', 'Late (16-30 days)'])]
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)

# =====================================================================
# 2. GEOGRAPHIC EDA & CHI-SQUARE TEST
# =====================================================================
print("\n" + "="*50)
print("--- GEOGRAPHIC EDA: ADDR_STATE CHI-SQUARE TEST ---")
print("="*50)

# Calculate default rates by state for context
state_stats = df.groupby('addr_state')['loan_status'].agg(['mean', 'count'])
state_stats.rename(columns={'mean': 'Default Rate (%)', 'count': 'Total Loans'}, inplace=True)
state_stats['Default Rate (%)'] = (state_stats['Default Rate (%)'] * 100).round(2)

print("Top 5 States with the Highest Default Rates:")
print(state_stats.sort_values(by='Default Rate (%)', ascending=False).head(5))

# Run the official Chi-Square Test
contingency_table_state = pd.crosstab(df['addr_state'], df['loan_status'])
chi2_stat_state, p_val_state, _, _ = chi2_contingency(contingency_table_state)

print(f"\nPearson Chi-Square statistic: {chi2_stat_state:.2f}")
print(f"Pearson Chi-Square p-value for 'addr_state': {p_val_state:.4f}")

if p_val_state < 0.05:
	print("\nConclusion: The map is statistically valid! There is a highly significant relationship between State and Default Risk.")
