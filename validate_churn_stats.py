import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('content/docs/project/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Basic info
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# 1. Verify churn distribution
print("\n===== CLAIM: 73.5% not churned, 26.5% churned =====")
churn_dist = df['Churn'].value_counts(normalize=True) * 100
print("Actual churn distribution:")
print(churn_dist)
print(f"Not Churned: {churn_dist['No']:.1f}%, Churned: {churn_dist['Yes']:.1f}%")

# 2. Verify contract type claims
print("\n===== CLAIM: Month-to-month (43%), one-year (11%), two-year (3%) churn rates =====")
contract_churn = df.groupby('Contract')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
print("Actual contract type churn rates:")
print(contract_churn)

# Create a visualization for contract type churn rates
plt.figure(figsize=(10, 6))
# Count the number of customers for each contract type and churn status
contract_counts = df.groupby(['Contract', 'Churn']).size().unstack()
# Calculate percentages
contract_percentages = contract_counts.div(contract_counts.sum(axis=1), axis=0) * 100

# Plot the data
ax = contract_percentages.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['No Churn', 'Churn'])

# Add value labels on top of bars
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            f'{height:.1f}%', 
            horizontalalignment='center', 
            verticalalignment='center')

plt.savefig('contract_churn_validation.png')
print("Contract churn plot saved to 'contract_churn_validation.png'")

# Verify each contract type claim
month_to_month_churn = None
one_year_churn = None
two_year_churn = None

for _, row in contract_churn.iterrows():
    if row['Contract'] == 'Month-to-month':
        month_to_month_churn = row['Churn']
    elif row['Contract'] == 'One year':
        one_year_churn = row['Churn']
    elif row['Contract'] == 'Two year':
        two_year_churn = row['Churn']

if month_to_month_churn is not None:
    print(f"Month-to-month churn rate: {month_to_month_churn:.1f}%")
    print(f"CLAIM: 43% - {'CORRECT' if abs(month_to_month_churn - 43) <= 2 else 'INCORRECT'}")

if one_year_churn is not None:
    print(f"One year churn rate: {one_year_churn:.1f}%")
    print(f"CLAIM: 11% - {'CORRECT' if abs(one_year_churn - 11) <= 2 else 'INCORRECT'}")

if two_year_churn is not None:
    print(f"Two year churn rate: {two_year_churn:.1f}%")
    print(f"CLAIM: 3% - {'CORRECT' if abs(two_year_churn - 3) <= 2 else 'INCORRECT'}")

# 3. Verify internet service claims
print("\n===== CLAIM: Fiber optic customers churn at nearly double the rate (42%) of DSL customers (19%) =====")
internet_churn = df.groupby('InternetService')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
print("Actual internet service churn rates:")
print(internet_churn)

# Create a visualization for internet service churn rates
plt.figure(figsize=(10, 6))
# Count the number of customers for each service type and churn status
internet_counts = df.groupby(['InternetService', 'Churn']).size().unstack()
# Calculate percentages
internet_percentages = internet_counts.div(internet_counts.sum(axis=1), axis=0) * 100

# Plot the data
ax = internet_percentages.plot(kind='bar', stacked=False, figsize=(10, 6), color=['green', 'red'])
plt.title('Churn Rate by Internet Service Type')
plt.xlabel('Internet Service Type')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['No Churn', 'Churn'])

# Add value labels on top of bars
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.text(x+width/2, 
            y+height/2, 
            f'{height:.1f}%', 
            horizontalalignment='center', 
            verticalalignment='center')

plt.savefig('internet_service_churn_validation.png')
print("Internet service churn plot saved to 'internet_service_churn_validation.png'")

# Calculate the ratio of Fiber optic to DSL churn rates
if 'Fiber optic' in internet_churn['InternetService'].values and 'DSL' in internet_churn['InternetService'].values:
    fiber_churn = internet_churn[internet_churn['InternetService'] == 'Fiber optic']['Churn'].values[0]
    dsl_churn = internet_churn[internet_churn['InternetService'] == 'DSL']['Churn'].values[0]
    ratio = fiber_churn / dsl_churn
    print(f"Fiber optic churn rate: {fiber_churn:.1f}%")
    print(f"DSL churn rate: {dsl_churn:.1f}%")
    print(f"Ratio of Fiber optic to DSL churn: {ratio:.1f}x")
    print(f"CLAIM: Fiber optic (42%) vs DSL (19%) - {'CORRECT' if abs(fiber_churn - 42) <= 2 and abs(dsl_churn - 19) <= 2 else 'INCORRECT'}")
    print(f"CLAIM: Nearly double rate - {'CORRECT' if 1.8 <= ratio <= 2.3 else 'INCORRECT'}")

# 4. Create plot to verify tenure distribution
print("\n===== Verifying tenure distribution claims =====")
plt.figure(figsize=(10, 6))
sns.histplot(
    data=df, x='tenure', 
    hue='Churn', multiple='dodge', 
    bins=12, palette=['green', 'red']
)
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.savefig('tenure_distribution_validation.png')
print("Tenure distribution plot saved to 'tenure_distribution_validation.png'")

# Calculate median tenure for churned and non-churned customers
median_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].median()
median_tenure_not_churned = df[df['Churn'] == 'No']['tenure'].median()
print(f"Median tenure for churned customers: {median_tenure_churned:.0f} months")
print(f"Median tenure for non-churned customers: {median_tenure_not_churned:.0f} months")
print(f"CLAIM: 10 months (churned) vs 38 months (loyal) - {'CORRECT' if abs(median_tenure_churned - 10) <= 1 and abs(median_tenure_not_churned - 38) <= 1 else 'INCORRECT'}")

# 5. Verify technical support churn claim
print("\n===== CLAIM: Customers with technical support are 33% less likely to churn =====")
tech_support_churn = df.groupby('TechSupport')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
print("Actual tech support churn rates:")
print(tech_support_churn)

if 'Yes' in tech_support_churn['TechSupport'].values and 'No' in tech_support_churn['TechSupport'].values:
    yes_churn = tech_support_churn[tech_support_churn['TechSupport'] == 'Yes']['Churn'].values[0]
    no_churn = tech_support_churn[tech_support_churn['TechSupport'] == 'No']['Churn'].values[0]
    reduction = ((no_churn - yes_churn) / no_churn) * 100
    print(f"Reduction in churn with tech support: {reduction:.1f}%")
    print(f"CLAIM: 33% less likely to churn - {'CORRECT' if abs(reduction - 33) <= 3 else 'INCORRECT'}")

# 6. Verify senior citizen claim
print("\n===== CLAIM: Senior citizens churn at a substantially higher rate (+19%) than non-seniors =====")
senior_churn = df.groupby('SeniorCitizen')['Churn'].apply(
    lambda x: (x == 'Yes').mean() * 100
).reset_index()
print("Actual senior citizen churn rates:")
print(senior_churn)

# Convert SeniorCitizen to more readable format for display
senior_churn['SeniorCitizen'] = senior_churn['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
print(senior_churn)

if 0 in df['SeniorCitizen'].values and 1 in df['SeniorCitizen'].values:
    senior_yes_churn = df[df['SeniorCitizen'] == 1]['Churn'].apply(lambda x: x == 'Yes').mean() * 100
    senior_no_churn = df[df['SeniorCitizen'] == 0]['Churn'].apply(lambda x: x == 'Yes').mean() * 100
    difference = senior_yes_churn - senior_no_churn
    print(f"Difference in churn rate: {difference:.1f}%")
    print(f"CLAIM: +19% higher churn rate - {'CORRECT' if abs(difference - 19) <= 2 else 'INCORRECT'}")

print("\nValidation complete.") 