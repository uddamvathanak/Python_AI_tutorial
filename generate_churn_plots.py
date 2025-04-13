import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output directory if it doesn't exist
output_dir = 'static/images/project/churn'
os.makedirs(output_dir, exist_ok=True)
print(f"Created output directory at {output_dir}")

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
df = pd.DataFrame({
    'tenure': np.random.randint(1, 73, n_samples),
    'monthly_charges': np.random.uniform(20, 120, n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2]),
    'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples)
})

# Calculate total charges with some random variation
df['total_charges'] = df['tenure'] * df['monthly_charges'] * (1 + np.random.normal(0, 0.1, n_samples))

# Generate churn values with higher probability for certain conditions
conditions = [
    (df['tenure'] < 12) & (df['monthly_charges'] > 80),
    (df['contract'] == 'Month-to-month'),
    (df['internet_service'] == 'Fiber optic')
]
base_prob = 0.15
additional_prob = np.zeros(n_samples)
for condition in conditions:
    additional_prob[condition] += 0.1
churn_prob = base_prob + additional_prob
df['churn'] = np.random.binomial(1, churn_prob)
df['churn'] = df['churn'].map({1: 'Yes', 0: 'No'})

try:
    # 1. Churn Distribution Pie Chart
    plt.figure(figsize=(8, 8))
    churn_counts = df['churn'].value_counts()
    plt.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
    plt.title('Customer Churn Distribution')
    plt.savefig(os.path.join(output_dir, 'churn_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated churn distribution pie chart")

    # 2. Tenure Distribution Box Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='churn', y='tenure', data=df)
    plt.title('Tenure Distribution by Churn Status')
    plt.xlabel('Churn')
    plt.ylabel('Tenure (months)')
    plt.savefig(os.path.join(output_dir, 'tenure_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated tenure distribution box plot")

    # 3. Monthly Charges vs Tenure Scatter Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='tenure', y='monthly_charges', hue='churn', alpha=0.6)
    plt.title('Monthly Charges vs Tenure')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Monthly Charges ($)')
    plt.savefig(os.path.join(output_dir, 'charges_vs_tenure.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated charges vs tenure scatter plot")

    # 4. Churn Rate by Internet Service
    plt.figure(figsize=(10, 6))
    churn_by_internet = df.groupby('internet_service')['churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    churn_by_internet.plot(kind='bar')
    plt.title('Churn Rate by Internet Service Type')
    plt.xlabel('Internet Service Type')
    plt.ylabel('Churn Rate (%)')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'churn_by_internet.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated churn by internet service bar plot")

    # 5. Contract Distribution Stacked Bar Plot
    plt.figure(figsize=(12, 6))
    contract_churn = pd.crosstab(df['contract'], df['churn'], normalize='index') * 100
    contract_churn.plot(kind='bar', stacked=True)
    plt.title('Contract Type Distribution by Churn Status')
    plt.xlabel('Contract Type')
    plt.ylabel('Percentage')
    plt.legend(title='Churn')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(output_dir, 'contract_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated contract distribution stacked bar plot")

    print("\nAll plots generated successfully!")

except Exception as e:
    print(f"Error generating plots: {str(e)}") 