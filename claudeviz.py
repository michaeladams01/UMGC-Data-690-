"""
DATA 690 Capstone - Loan Default Prediction
Units 4-5: Data Selection & Descriptive Analysis + Data Visualizations
Author: Michael A. Adams

This script covers everything completed through the "Data Visualizations" write-up:
  1. Data acquisition / load
  2. Target variable formulation (Fully Paid -> 0, Charged Off -> 1)
  3. Sparsity management (drop columns missing > 50% of data)
  4. Descriptive statistics for core continuous features
  5. Data Visualization 1: Pearson correlation heatmap
  6. Data Visualization 2: Box plot of interest rate by loan status + Welch's t-test
  7. Data Visualization 3: Stacked bar chart of loan volume/defaults by grade + chi-square test

NOTE: All EDA here is run on the RAW, pre-split dataset for sanity-checking purposes
only (distributions, class balance, obvious data issues). Any descriptive statistics
used to inform feature engineering decisions for modeling should be recomputed on the
TRAINING split only, after train/test split, to avoid data leakage.
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_style("whitegrid")

# -----------------------------------------------------------------------
# 1. Data Acquisition
# -----------------------------------------------------------------------
# Update this path to wherever the LendingClub CSV lives in your environment
# (local path, Google Drive mount, or Colab upload).
DATA_PATH = r"C:\Users\Adams\Downloads\Model\loan.csv"

df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Raw shape: {df_raw.shape}")

# -----------------------------------------------------------------------
# 2. Target Variable Formulation
# -----------------------------------------------------------------------
# Only loans with a terminal status are usable for a binary default classifier.
terminal_statuses = ["Fully Paid", "Charged Off"]
df = df_raw[df_raw["loan_status"].isin(terminal_statuses)].copy()

df["loan_status_binary"] = df["loan_status"].map(
    {"Fully Paid": 0, "Charged Off": 1}
)

print(f"Shape after filtering to terminal loans: {df.shape}")

status_counts = df["loan_status_binary"].value_counts().sort_index()
status_pct = df["loan_status_binary"].value_counts(normalize=True).sort_index()
status_summary = pd.DataFrame({"Count": status_counts, "Proportion": status_pct.round(4)})
status_summary.index = ["Fully Paid (0)", "Charged Off (1)"]
print("\nTarget variable distribution:")
print(status_summary)

# -----------------------------------------------------------------------
# 3. Sparsity and Missing Data Management
# -----------------------------------------------------------------------
missing_frac = df.isna().mean().sort_values(ascending=False)
cols_to_drop = missing_frac[missing_frac > 0.50].index.tolist()

print(f"Columns dropped (>50% missing): {len(cols_to_drop)} of {df.shape[1]}")
df = df.drop(columns=cols_to_drop)
print(f"Shape after sparsity thresholding: {df.shape}")

# -----------------------------------------------------------------------
# 4. Descriptive Statistics (core continuous financial variables)
# -----------------------------------------------------------------------
core_vars = ["loan_amnt", "int_rate", "annual_inc", "dti"]
desc_stats = df[core_vars].agg(["min", "max", "mean", "std"]).T
desc_stats.columns = ["Minimum", "Maximum", "Mean", "Standard Deviation"]
print("\nDescriptive Statistics:")
print(desc_stats.round(2))

# Stratify means against the target to confirm the baseline credit-risk story
strat_means = df.groupby("loan_status_binary")[core_vars].mean().round(2)
print("\nMeans by loan_status_binary (0=Fully Paid, 1=Charged Off):")
print(strat_means)

# -----------------------------------------------------------------------
# 4b. Supplemental Visualization: Class Imbalance (target variable)
# -----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
ax = status_counts.rename(index={0: "Fully Paid (0)", 1: "Charged Off (1)"}).plot(
    kind="bar", color=["#4C9A8E", "#E07B54"]
)
for i, v in enumerate(status_counts.values):
    ax.text(i, v + 5000, f"{v:,}\n({status_pct.values[i]*100:.1f}%)",
            ha="center", va="bottom", fontsize=10)
plt.title("Class Distribution of Target Variable (loan_status_binary)")
plt.xlabel("Loan Resolution Status")
plt.ylabel("Total Count of Loans")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("viz0_class_imbalance.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# 4c. Supplemental Visualization: Missing Data by Column (pre-drop)
# -----------------------------------------------------------------------
# Recomputed against the raw, pre-filter dataframe to show the actual
# missingness landscape that justified the 50% drop threshold.
missing_frac_raw = df_raw.isna().mean().sort_values(ascending=False)
top_missing = missing_frac_raw.head(20)

plt.figure(figsize=(8, 8))
colors = ["#B03A2E" if v > 0.50 else "#5DADE2" for v in top_missing.values]
plt.barh(top_missing.index[::-1], top_missing.values[::-1] * 100, color=colors[::-1])
plt.axvline(50, color="black", linestyle="--", linewidth=1, label="50% drop threshold")
plt.xlabel("Percent Missing (%)")
plt.title("Top 20 Columns by Missing Data Percentage")
plt.legend()
plt.tight_layout()
plt.savefig("viz0b_missingness.png", dpi=150)
plt.show()

print(f"\nColumns exceeding 50% missing (dropped): {(missing_frac_raw > 0.50).sum()} of {df_raw.shape[1]}")

# -----------------------------------------------------------------------
# 5. Data Visualization 1: Pearson Correlation Heatmap
# -----------------------------------------------------------------------
heatmap_vars = ["loan_amnt", "int_rate", "annual_inc", "dti", "installment"]
corr_matrix = df[heatmap_vars].corr(method="pearson")

plt.figure(figsize=(7, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={"label": "Pearson r"},
)
plt.title("Pearson Correlation Heatmap of Primary Financial Features")
plt.tight_layout()
plt.savefig("viz1_correlation_heatmap.png", dpi=150)
plt.show()

print(f"\nloan_amnt vs installment correlation: {corr_matrix.loc['loan_amnt', 'installment']:.3f}")

# -----------------------------------------------------------------------
# 6. Data Visualization 2: Box Plot of Interest Rate by Loan Status
# -----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.boxplot(
    data=df,
    x="loan_status_binary",
    y="int_rate",
    hue="loan_status_binary",
    palette={0: "#4C9A8E", 1: "#E07B54"},
    legend=False,
)
plt.xticks([0, 1], ["Fully Paid (0)", "Charged Off (1)"])
plt.xlabel("Loan Resolution Status")
plt.ylabel("Interest Rate (%)")
plt.title("Distribution of Interest Rates by Loan Status")
plt.tight_layout()
plt.savefig("viz2_boxplot_interest_rate.png", dpi=150)
plt.show()

# Welch's Two-Sample T-Test (unequal variances assumed)
paid = df.loc[df["loan_status_binary"] == 0, "int_rate"].dropna()
charged_off = df.loc[df["loan_status_binary"] == 1, "int_rate"].dropna()
t_stat, p_val = stats.ttest_ind(charged_off, paid, equal_var=False)

print(f"\nMedian int_rate - Fully Paid: {paid.median():.2f}%")
print(f"Median int_rate - Charged Off: {charged_off.median():.2f}%")
print(f"Welch's t-test: t = {t_stat:.3f}, p = {p_val:.4g}")

# -----------------------------------------------------------------------
# 6b. Supplemental Visualization: Box Plot of DTI by Loan Status
# -----------------------------------------------------------------------
plt.figure(figsize=(7, 6))
sns.boxplot(
    data=df,
    x="loan_status_binary",
    y="dti",
    hue="loan_status_binary",
    palette={0: "#4C9A8E", 1: "#E07B54"},
    legend=False,
)
plt.ylim(-5, 50)  # dti has extreme outliers (max 999); clip for readability
plt.xticks([0, 1], ["Fully Paid (0)", "Charged Off (1)"])
plt.xlabel("Loan Resolution Status")
plt.ylabel("Debt-to-Income Ratio")
plt.title("Distribution of Debt-to-Income Ratio by Loan Status")
plt.tight_layout()
plt.savefig("viz2b_boxplot_dti.png", dpi=150)
plt.show()

dti_paid = df.loc[df["loan_status_binary"] == 0, "dti"].dropna()
dti_charged_off = df.loc[df["loan_status_binary"] == 1, "dti"].dropna()
t_stat_dti, p_val_dti = stats.ttest_ind(dti_charged_off, dti_paid, equal_var=False)
print(f"\nMedian DTI - Fully Paid: {dti_paid.median():.2f}")
print(f"Median DTI - Charged Off: {dti_charged_off.median():.2f}")
print(f"Welch's t-test (DTI): t = {t_stat_dti:.3f}, p = {p_val_dti:.4g}")

# -----------------------------------------------------------------------
# 7. Data Visualization 3: Stacked Bar Chart of Loan Grade vs Default Status
# -----------------------------------------------------------------------
grade_status = pd.crosstab(df["grade"], df["loan_status_binary"])
grade_status = grade_status.reindex(sorted(grade_status.index))  # A-G order
grade_status.columns = ["Fully Paid (0)", "Charged Off (1)"]

ax = grade_status.plot(
    kind="bar",
    stacked=True,
    color=["#F4D03F", "#4A235A"],
    figsize=(8, 6),
)
plt.title("Loan Volume and Default Proportion by Assigned Loan Grade")
plt.xlabel("Lending Club Assigned Loan Grade (A=Prime, G=Subprime)")
plt.ylabel("Total Count of Loans")
plt.legend(title="Loan Status")
plt.tight_layout()
plt.savefig("viz3_stacked_bar_grade.png", dpi=150)
plt.show()

# Pearson Chi-Square Test of Independence: grade vs default status
chi2, chi_p, dof, expected = stats.chi2_contingency(grade_status)
print(f"\nChi-Square test (grade vs default): chi2 = {chi2:.2f}, p = {chi_p:.4g}, dof = {dof}")

print("\nEDA through Data Visualizations complete.")

#-----------------------------------------------------------------------
# 8. Variable Selection Summary (post spartity-threshold, pre-feature engineering)
#-----------------------------------------------------------------------
print(f"\nTotal columns remaining after 50% missing drop: {df.shape[1]}" )
print("\nColumns remaining:" )
for col in df.columns:
    print(f" - {col} ({df[col].dtype})")

X_numeric = df[numeric_candidate_cols].dropna()
vif_data = pd.DataFrame()
vif_data["feature"] = X_numeric.columns
vif_data["VIF"] = [variance_inflation_factor(X_numeric.values, i) for i in range(X_numeric.shape[1])]
print(vif_data.sort_values("VIF", ascending=False))