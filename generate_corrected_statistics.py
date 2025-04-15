import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('content/docs/project/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print(f"Dataset shape: {df.shape}")
print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")

# VERIFICATION 1: Churn Distribution
print("\n===== CHURN DISTRIBUTION =====")
churn_counts = df['Churn'].value_counts()
churn_percent = df['Churn'].value_counts(normalize=True) * 100
print("Count:")
print(churn_counts)
print("\nPercentage:")
print(churn_percent)

# VERIFICATION 2: Internet Service Churn Rates
print("\n===== INTERNET SERVICE CHURN RATES =====")
internet_cross = pd.crosstab(df['InternetService'], df['Churn'])
print("Counts:")
print(internet_cross)

internet_percent = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
print("\nPercentage:")
print(internet_percent)

# Calculate the ratio of Fiber optic to DSL churn
fiber_churn = internet_percent.loc['Fiber optic', 'Yes'] if 'Fiber optic' in internet_percent.index and 'Yes' in internet_percent.columns else 0
dsl_churn = internet_percent.loc['DSL', 'Yes'] if 'DSL' in internet_percent.index and 'Yes' in internet_percent.columns else 0
no_internet_churn = internet_percent.loc['No', 'Yes'] if 'No' in internet_percent.index and 'Yes' in internet_percent.columns else 0

if fiber_churn > 0 and dsl_churn > 0:
    ratio = fiber_churn / dsl_churn
    print(f"\nFiber optic churn rate: {fiber_churn:.1f}%")
    print(f"DSL churn rate: {dsl_churn:.1f}%")
    print(f"No internet churn rate: {no_internet_churn:.1f}%")
    print(f"Ratio of Fiber optic to DSL churn: {ratio:.2f}x")

# VERIFICATION 3: Contract Type Churn Rates
print("\n===== CONTRACT TYPE CHURN RATES =====")
contract_cross = pd.crosstab(df['Contract'], df['Churn'])
print("Counts:")
print(contract_cross)

contract_percent = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print("\nPercentage:")
print(contract_percent)

# VERIFICATION 4: Tenure Statistics
print("\n===== TENURE STATISTICS =====")
median_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].median()
median_tenure_not_churned = df[df['Churn'] == 'No']['tenure'].median()
print(f"Median tenure for churned customers: {median_tenure_churned:.1f} months")
print(f"Median tenure for non-churned customers: {median_tenure_not_churned:.1f} months")

# VERIFICATION 5: Technical Support Impact
print("\n===== TECHNICAL SUPPORT IMPACT =====")
tech_support_cross = pd.crosstab(df['TechSupport'], df['Churn'])
print("Counts:")
print(tech_support_cross)

tech_support_percent = pd.crosstab(df['TechSupport'], df['Churn'], normalize='index') * 100
print("\nPercentage:")
print(tech_support_percent)

yes_support_churn = tech_support_percent.loc['Yes', 'Yes'] if 'Yes' in tech_support_percent.index and 'Yes' in tech_support_percent.columns else 0
no_support_churn = tech_support_percent.loc['No', 'Yes'] if 'No' in tech_support_percent.index and 'Yes' in tech_support_percent.columns else 0

if yes_support_churn > 0 and no_support_churn > 0:
    reduction = ((no_support_churn - yes_support_churn) / no_support_churn) * 100
    print(f"\nWith tech support churn rate: {yes_support_churn:.1f}%")
    print(f"Without tech support churn rate: {no_support_churn:.1f}%")
    print(f"Reduction in churn with tech support: {reduction:.1f}%")

# VERIFICATION 6: Senior Citizens
print("\n===== SENIOR CITIZEN IMPACT =====")
df['SeniorCitizenStr'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
senior_cross = pd.crosstab(df['SeniorCitizenStr'], df['Churn'])
print("Counts:")
print(senior_cross)

senior_percent = pd.crosstab(df['SeniorCitizenStr'], df['Churn'], normalize='index') * 100
print("\nPercentage:")
print(senior_percent)

senior_yes_churn = senior_percent.loc['Yes', 'Yes'] if 'Yes' in senior_percent.index and 'Yes' in senior_percent.columns else 0
senior_no_churn = senior_percent.loc['No', 'Yes'] if 'No' in senior_percent.index and 'Yes' in senior_percent.columns else 0

if 'Yes' in senior_percent.index and 'No' in senior_percent.index:
    difference = senior_yes_churn - senior_no_churn
    print(f"\nSenior citizens churn rate: {senior_yes_churn:.1f}%")
    print(f"Non-senior citizens churn rate: {senior_no_churn:.1f}%")
    print(f"Difference: +{difference:.1f}%")

print("\nVerification complete. Please check these statistics against the markdown.") 