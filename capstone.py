import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind 
from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

print("All libraries imported successfully!")
filepath = "C:\\Users\\Adams\\Downloads\\Model\\loan.csv"
df = pd.read_csv(filepath, low_memory=False)
# Right after read_csv
print(df.shape)
print(df.dtypes.value_counts())
print(df.isnull().mean().sort_values(ascending=False).head(20))
print(df['loan_status'].value_counts())

# =====================================================================
# 1. INITIAL CLEANING & TARGET ENCODING
# =====================================================================
# Initial dropping of irrelevant, leaky, or sparse columns
df = df.drop(columns=['id', 'member_id', 'url', 'desc', 'title', 'zip_code', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 
	'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 
	'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 
	'hardship_flag', 'debt_settlement_flag', 'funded_amnt', 'funded_amnt_inv',
	'issue_d','sub_grade', 'pymnt_plan'], errors='ignore')

# Drops columns missing more than 50% of their data
df = df.dropna(thresh=int(0.5 * len(df)), axis=1)

# Define and clean Target Variable
bad_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off', 'Late (31-120 days)']
df = df[~df['loan_status'].isin(['Current', 'In Grace Period', 'Late (16-30 days)'])]
df['loan_status'] = df['loan_status'].apply(lambda x: 1 if x in bad_statuses else 0)

# =====================================================================
# 2. OPTION 4: TRAIN / TEST SPLIT
# =====================================================================
X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.20, stratify=y, random_state=42
)

# Recombine training data for analysis
train_data = pd.concat([X_train, y_train], axis=1)

# =====================================================================
# 3. DESCRIPTIVE STATISTICS (For your Word Template)
# =====================================================================
print("\n" + "="*50)
print("--- DESCRIPTIVE STATISTICS FOR PROJECT TEMPLATE ---")
print("="*50)

print(f"Total historical loans analyzed: {len(df):,}")
print(f"Loans allocated to Training (80%): {len(train_data):,}")
print(f"Loans allocated to Testing (20%): {len(X_test):,}")

default_count = train_data['loan_status'].sum()
default_rate = (default_count / len(train_data)) * 100
print(f"\nTraining Set Default Rate: {default_rate:.2f}% ({default_count:,} defaults)")

# Summary of key continuous variables
key_numeric_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 'fico_range_low']
available_num_cols = [col for col in key_numeric_cols if col in train_data.columns]

print("\n--- Overall Summary of Key Financial Variables ---")
print(train_data[available_num_cols].describe().round(2).loc[['mean', 'std', 'min', '50%', 'max']])

print("\n--- Stratified Means: Default (1) vs Fully Paid (0) ---")
stratified_means = train_data.groupby('loan_status')[available_num_cols].mean().round(2)
print(stratified_means.T)
print("="*50 + "\n")

# =====================================================================
# 4. AUTOMATED UNIVARIABLE SCREENING (Purposeful Selection)
# =====================================================================
print("Starting Automated Univariable Screening (This takes a moment)...")

candidate_vars = []
rejected_vars = []

# Screen Categorical (Chi-Square)
cat_cols = train_data.select_dtypes(include=['object', 'str']).columns
for col in cat_cols:
	if col == 'loan_status': continue
	contingency_table = pd.crosstab(train_data[col], train_data['loan_status'])
	if contingency_table.shape[0] < 2:
		rejected_vars.append(col)
		continue
	chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
	if p_val < 0.25:
		candidate_vars.append((col, 'Categorical', p_val))
	else:
		rejected_vars.append(col)

# Screen Continuous (Logistic Regression)
num_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
	if col == 'loan_status': continue
	temp_data = train_data[[col, 'loan_status']].dropna()
	if len(temp_data) == 0: continue
	
	X_cont = sm.add_constant(temp_data[col])
	y_cont = temp_data['loan_status']
	try:
		model = sm.Logit(y_cont, X_cont).fit(disp=0)
		if model.pvalues[col] < 0.25:
			candidate_vars.append((col, 'Continuous', model.pvalues[col]))
		else:
			rejected_vars.append(col)
	except:
		rejected_vars.append(col)

candidates_df = pd.DataFrame(candidate_vars, columns=['Variable', 'Type', 'P-Value'])
print(f"\nScreening Complete! Kept {len(candidates_df)} candidates. Rejected {len(rejected_vars)} variables.")

# =====================================================================
# 5. MULTIVARIABLE MODEL PREPARATION
# =====================================================================
surviving_vars = candidates_df['Variable'].tolist()
X_candidates = train_data[surviving_vars]

# Filter High Cardinality and Leaky dates to prevent memory crash
high_cardinality_cols = [col for col in X_candidates.select_dtypes(include=['object', 'str']).columns if X_candidates[col].nunique() > 50]
manual_drops = ['last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line', 'issue_d', 'addr_state']
cols_to_drop = list(set(high_cardinality_cols + manual_drops))

final_surviving_vars = [var for var in surviving_vars if var not in cols_to_drop]
X_candidates_clean = train_data[final_surviving_vars].copy()
y_train = train_data['loan_status']

print(f"\nFeatures remaining for Multivariable Model: {len(final_surviving_vars)}")

# --- CRITICAL IMPUTATION STEP ---
print("Imputing missing values to prevent model crash...")
num_cols_clean = X_candidates_clean.select_dtypes(include=['int64', 'float64']).columns
cat_cols_clean = X_candidates_clean.select_dtypes(include=['object', 'str']).columns

for col in num_cols_clean:
	X_candidates_clean[col] = X_candidates_clean[col].fillna(X_candidates_clean[col].median())
for col in cat_cols_clean:
	X_candidates_clean[col] = X_candidates_clean[col].fillna('Missing')

# Encode text into 1s and 0s
X_encoded = pd.get_dummies(X_candidates_clean, drop_first=True).astype(float)
X_encoded = sm.add_constant(X_encoded) 

# =====================================================================
# 6. FIT INITIAL MULTIVARIABLE MODEL
# =====================================================================
print("\nFitting the Multivariable Model... (This will take a minute or two)")
try:
	multi_model = sm.Logit(y_train, X_encoded).fit(disp=0)
	print("\n--- INITIAL MULTIVARIABLE MODEL SUMMARY ---")
	print(multi_model.summary())
except Exception as e:
	print(f"Model fitting failed: {e}")
