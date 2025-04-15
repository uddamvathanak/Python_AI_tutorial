"""
Churn Prediction Documentation Validator

This script validates the key claims made in the churn prediction documentation against the actual data.
It checks percentages, statistics, and generates validation charts to verify the accuracy of the documentation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directory for images
os.makedirs('validation_results', exist_ok=True)

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('content/docs/project/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Print basic dataset info
print(f"Dataset shape: {df.shape}")
print(f"Missing values after preprocessing: {df.isnull().sum().sum()}")

# Create a results summary list to track accuracy
results = []

# 1. Validate overall churn distribution
print("\n===== Validating overall churn distribution =====")
print("CLAIM: 73.5% not churned, 26.5% churned")

churn_dist = df['Churn'].value_counts(normalize=True) * 100
not_churned = churn_dist['No']
churned = churn_dist['Yes']

print(f"ACTUAL: {not_churned:.1f}% not churned, {churned:.1f}% churned")
churn_accuracy = abs(not_churned - 73.5) <= 1 and abs(churned - 26.5) <= 1
print(f"ASSESSMENT: {'CORRECT' if churn_accuracy else 'INCORRECT'}")
results.append(("Overall churn distribution", churn_accuracy))

# Plot the actual distribution
plt.figure(figsize=(10, 6))
ax = sns.countplot(x='Churn', data=df, palette=['green', 'red'])
plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Count')

# Add percentages above bars
total = len(df)
for p in ax.patches:
    height = p.get_height()
    percentage = 100 * height / total
    ax.text(p.get_x() + p.get_width()/2, height + 50, f'{percentage:.1f}%', 
            ha='center', fontsize=12)

plt.savefig('validation_results/churn_distribution.png')
print("Churn distribution plot saved")

# 2. Validate contract type churn rates
print("\n===== Validating contract type churn rates =====")
print("CLAIM: Month-to-month (43%), one-year (11%), two-year (3%) churn rates")

contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
month_churn = contract_churn.loc['Month-to-month', 'Yes']
one_year_churn = contract_churn.loc['One year', 'Yes']
two_year_churn = contract_churn.loc['Two year', 'Yes']

print(f"ACTUAL: Month-to-month ({month_churn:.1f}%), one-year ({one_year_churn:.1f}%), two-year ({two_year_churn:.1f}%)")
month_accurate = abs(month_churn - 43) <= 1
one_year_accurate = abs(one_year_churn - 11) <= 1
two_year_accurate = abs(two_year_churn - 3) <= 1
contract_accuracy = month_accurate and one_year_accurate and two_year_accurate
print(f"ASSESSMENT: {'CORRECT' if contract_accuracy else 'INCORRECT'}")
results.append(("Contract churn rates", contract_accuracy))

# Plot contract type churn
plt.figure(figsize=(12, 7))
contract_plot = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
ax = contract_plot.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Percentage (%)')
plt.legend(['No Churn', 'Churn'])

# Add percentages to bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if i >= len(ax.patches)/2:  # Only label the 'Yes' bars
        ax.text(x+width/2, height/2 + y, f'{height:.1f}%', ha='center')

plt.savefig('validation_results/contract_churn.png')
print("Contract type churn plot saved")

# 3. Validate internet service churn rates
print("\n===== Validating internet service churn rates =====")
print("CLAIM: Fiber optic customers churn at nearly double the rate (42%) of DSL customers (19%)")

internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
fiber_churn = internet_churn.loc['Fiber optic', 'Yes'] if 'Fiber optic' in internet_churn.index else 0
dsl_churn = internet_churn.loc['DSL', 'Yes'] if 'DSL' in internet_churn.index else 0
no_internet_churn = internet_churn.loc['No', 'Yes'] if 'No' in internet_churn.index else 0

print(f"ACTUAL: Fiber optic ({fiber_churn:.1f}%), DSL ({dsl_churn:.1f}%)")
if fiber_churn > 0 and dsl_churn > 0:
    ratio = fiber_churn / dsl_churn
    print(f"ACTUAL RATIO: {ratio:.2f}x")
    fiber_accurate = abs(fiber_churn - 42) <= 2
    dsl_accurate = abs(dsl_churn - 19) <= 2
    ratio_accurate = 1.8 <= ratio <= 2.3
    internet_accuracy = fiber_accurate and dsl_accurate and ratio_accurate
    print(f"ASSESSMENT: {'CORRECT' if internet_accuracy else 'INCORRECT'}")
    results.append(("Internet service churn rates", internet_accuracy))

# Plot internet service churn
plt.figure(figsize=(12, 7))
internet_plot = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
ax = internet_plot.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Internet Service Type')
plt.xlabel('Internet Service Type')
plt.ylabel('Percentage (%)')
plt.legend(['No Churn', 'Churn'])

# Add percentages to bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if i >= len(ax.patches)/2:  # Only label the 'Yes' bars
        ax.text(x+width/2, height/2 + y, f'{height:.1f}%', ha='center')

plt.savefig('validation_results/internet_service_churn.png')
print("Internet service churn plot saved")

# 4. Validate tenure distribution claim
print("\n===== Validating tenure distribution claims =====")
print("CLAIM: Median tenure for churned customers is 10 months vs 38 months for loyal customers")

median_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].median()
median_tenure_not_churned = df[df['Churn'] == 'No']['tenure'].median()

print(f"ACTUAL: Median tenure for churned customers is {median_tenure_churned:.1f} months vs {median_tenure_not_churned:.1f} months for loyal customers")
tenure_accurate = abs(median_tenure_churned - 10) <= 1 and abs(median_tenure_not_churned - 38) <= 1
print(f"ASSESSMENT: {'CORRECT' if tenure_accurate else 'INCORRECT'}")
results.append(("Tenure distribution claim", tenure_accurate))

# Plot tenure distribution
plt.figure(figsize=(12, 7))
ax = sns.histplot(data=df, x='tenure', hue='Churn', multiple='dodge', bins=12, palette=['green', 'red'])
plt.axvline(x=median_tenure_churned, color='red', linestyle='--', label=f'Median Churned: {median_tenure_churned} months')
plt.axvline(x=median_tenure_not_churned, color='green', linestyle='--', label=f'Median Not Churned: {median_tenure_not_churned} months')
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.legend()
plt.savefig('validation_results/tenure_distribution.png')
print("Tenure distribution plot saved")

# 5. Validate technical support claim
print("\n===== Validating technical support churn reduction claim =====")
print("CLAIM: Customers with technical support are 33% less likely to churn")

tech_support_churn = pd.crosstab(df['TechSupport'], df['Churn'], normalize='index') * 100
yes_support_churn = tech_support_churn.loc['Yes', 'Yes'] if 'Yes' in tech_support_churn.index else 0
no_support_churn = tech_support_churn.loc['No', 'Yes'] if 'No' in tech_support_churn.index else 0

print(f"ACTUAL: With tech support: {yes_support_churn:.1f}% churn, Without: {no_support_churn:.1f}% churn")
if yes_support_churn > 0 and no_support_churn > 0:
    reduction = ((no_support_churn - yes_support_churn) / no_support_churn) * 100
    print(f"ACTUAL REDUCTION: {reduction:.1f}%")
    reduction_accurate = abs(reduction - 33) <= 3
    print(f"ASSESSMENT: {'CORRECT' if reduction_accurate else 'INCORRECT'}")
    results.append(("Tech support churn reduction", reduction_accurate))

# Plot technical support churn
plt.figure(figsize=(12, 7))
tech_plot = pd.crosstab(df['TechSupport'], df['Churn'], normalize='index') * 100
ax = tech_plot.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Technical Support')
plt.xlabel('Has Technical Support')
plt.ylabel('Percentage (%)')
plt.legend(['No Churn', 'Churn'])

# Add percentages to bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if i >= len(ax.patches)/2:  # Only label the 'Yes' bars
        ax.text(x+width/2, height/2 + y, f'{height:.1f}%', ha='center')

plt.savefig('validation_results/tech_support_churn.png')
print("Tech support churn plot saved")

# 6. Validate senior citizen claim
print("\n===== Validating senior citizen churn rate claim =====")
print("CLAIM: Senior citizens churn at a substantially higher rate (+19%) than non-seniors")

# Convert SeniorCitizen to string for better display
df['SeniorCitizenStr'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
senior_churn = pd.crosstab(df['SeniorCitizenStr'], df['Churn'], normalize='index') * 100
senior_yes_churn = senior_churn.loc['Yes', 'Yes'] if 'Yes' in senior_churn.index else 0
senior_no_churn = senior_churn.loc['No', 'Yes'] if 'No' in senior_churn.index else 0

difference = senior_yes_churn - senior_no_churn
print(f"ACTUAL: Senior citizens: {senior_yes_churn:.1f}%, Non-seniors: {senior_no_churn:.1f}%")
print(f"ACTUAL DIFFERENCE: +{difference:.1f}%")
senior_accurate = abs(difference - 19) <= 2
print(f"ASSESSMENT: {'CORRECT' if senior_accurate else 'INCORRECT'}")
results.append(("Senior citizen churn rate difference", senior_accurate))

# Plot senior citizen churn
plt.figure(figsize=(12, 7))
senior_plot = pd.crosstab(df['SeniorCitizenStr'], df['Churn'], normalize='index') * 100
ax = senior_plot.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Senior Citizen Status')
plt.xlabel('Senior Citizen')
plt.ylabel('Percentage (%)')
plt.legend(['No Churn', 'Churn'])

# Add percentages to bars
for i, p in enumerate(ax.patches):
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    if i >= len(ax.patches)/2:  # Only label the 'Yes' bars
        ax.text(x+width/2, height/2 + y, f'{height:.1f}%', ha='center')

plt.savefig('validation_results/senior_citizen_churn.png')
print("Senior citizen churn plot saved")

# Print overall assessment
print("\n===== OVERALL ASSESSMENT =====")
print(f"Total claims validated: {len(results)}")
correct_claims = sum(1 for _, accurate in results if accurate)
print(f"Correct claims: {correct_claims} ({correct_claims/len(results)*100:.1f}%)")
incorrect_claims = sum(1 for _, accurate in results if not accurate)
print(f"Incorrect claims: {incorrect_claims} ({incorrect_claims/len(results)*100:.1f}%)")

print("\nDetailed results:")
for claim, accurate in results:
    print(f"- {claim}: {'CORRECT' if accurate else 'INCORRECT'}")

print("\nValidation complete. Results saved in 'validation_results' directory") 