import pandas as pd
import numpy as np
import csv

print("Loading dataset...")
df = pd.read_csv('content/docs/project/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle missing values in TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Create a CSV to store all statistics
stats = []

# 1. Overall churn distribution
churn_dist = df['Churn'].value_counts(normalize=True) * 100
not_churned = churn_dist['No']
churned = churn_dist['Yes']
stats.append(["Overall churn distribution", 
              f"Not churned: {not_churned:.1f}%, Churned: {churned:.1f}%"])

# 2. Contract type churn rates
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
month_churn = contract_churn.loc['Month-to-month', 'Yes']
one_year_churn = contract_churn.loc['One year', 'Yes']
two_year_churn = contract_churn.loc['Two year', 'Yes']
stats.append(["Month-to-month contract churn rate", f"{month_churn:.1f}%"])
stats.append(["One-year contract churn rate", f"{one_year_churn:.1f}%"])
stats.append(["Two-year contract churn rate", f"{two_year_churn:.1f}%"])

# 3. Internet service churn rates
internet_churn = pd.crosstab(df['InternetService'], df['Churn'], normalize='index') * 100
fiber_churn = internet_churn.loc['Fiber optic', 'Yes'] if 'Fiber optic' in internet_churn.index else 0
dsl_churn = internet_churn.loc['DSL', 'Yes'] if 'DSL' in internet_churn.index else 0
no_internet_churn = internet_churn.loc['No', 'Yes'] if 'No' in internet_churn.index else 0
stats.append(["Fiber optic churn rate", f"{fiber_churn:.1f}%"])
stats.append(["DSL churn rate", f"{dsl_churn:.1f}%"])
stats.append(["No internet churn rate", f"{no_internet_churn:.1f}%"])
stats.append(["Fiber to DSL ratio", f"{fiber_churn/dsl_churn:.2f}x"])

# 4. Tenure statistics
median_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].median()
median_tenure_not_churned = df[df['Churn'] == 'No']['tenure'].median()
stats.append(["Median tenure (churned)", f"{median_tenure_churned:.1f} months"])
stats.append(["Median tenure (not churned)", f"{median_tenure_not_churned:.1f} months"])

# 5. Technical support impact
tech_support_churn = pd.crosstab(df['TechSupport'], df['Churn'], normalize='index') * 100
yes_support_churn = tech_support_churn.loc['Yes', 'Yes'] if 'Yes' in tech_support_churn.index else 0
no_support_churn = tech_support_churn.loc['No', 'Yes'] if 'No' in tech_support_churn.index else 0
reduction = ((no_support_churn - yes_support_churn) / no_support_churn) * 100
stats.append(["Tech support churn rate", f"{yes_support_churn:.1f}%"])
stats.append(["No tech support churn rate", f"{no_support_churn:.1f}%"])
stats.append(["Tech support churn reduction", f"{reduction:.1f}%"])

# 6. Senior citizen impact
df['SeniorCitizenStr'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
senior_churn = pd.crosstab(df['SeniorCitizenStr'], df['Churn'], normalize='index') * 100
senior_yes_churn = senior_churn.loc['Yes', 'Yes'] if 'Yes' in senior_churn.index else 0
senior_no_churn = senior_churn.loc['No', 'Yes'] if 'No' in senior_churn.index else 0
difference = senior_yes_churn - senior_no_churn
stats.append(["Senior citizens churn rate", f"{senior_yes_churn:.1f}%"])
stats.append(["Non-seniors churn rate", f"{senior_no_churn:.1f}%"])
stats.append(["Senior vs non-senior difference", f"+{difference:.1f}%"])

# Export to CSV
with open('churn_statistics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Statistic', 'Value'])
    writer.writerows(stats)

print("Statistics exported to churn_statistics.csv")

# Print for easy reference
print("\nKey Statistics for Churn Prediction Analysis:")
print("--------------------------------------------")
for stat, value in stats:
    print(f"{stat}: {value}")

# Additional check: Markdown values vs. actual values
markdown_values = {
    "Overall churn": {"markdown": "76.6% loyal, 23.4% churned", 
                      "actual": f"{not_churned:.1f}% loyal, {churned:.1f}% churned"},
    "Month-to-month contract": {"markdown": "42.7%", "actual": f"{month_churn:.1f}%"},
    "One-year contract": {"markdown": "11.3%", "actual": f"{one_year_churn:.1f}%"},
    "Two-year contract": {"markdown": "2.8%", "actual": f"{two_year_churn:.1f}%"},
    "Fiber optic": {"markdown": "41.9%", "actual": f"{fiber_churn:.1f}%"},
    "DSL": {"markdown": "19.0%", "actual": f"{dsl_churn:.1f}%"},
    "No internet": {"markdown": "7.4%", "actual": f"{no_internet_churn:.1f}%"},
    "Median tenure (churned)": {"markdown": "11 months", "actual": f"{median_tenure_churned:.0f} months"},
    "Median tenure (not churned)": {"markdown": "39 months", "actual": f"{median_tenure_not_churned:.0f} months"},
    "Tech support reduction": {"markdown": "34%", "actual": f"{reduction:.0f}%"},
    "Senior citizen difference": {"markdown": "+18.1%", "actual": f"+{difference:.1f}%"}
}

print("\nComparison of Markdown Values vs. Actual Values:")
print("-----------------------------------------------")
for key, val in markdown_values.items():
    match = "✓" if val["markdown"] == val["actual"] else "✗"
    print(f"{key} - Markdown: {val['markdown']}, Actual: {val['actual']} {match}") 