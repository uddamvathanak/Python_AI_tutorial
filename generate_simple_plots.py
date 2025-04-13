import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

# Create directory if needed
os.makedirs('static/images/project/churn', exist_ok=True)

# 1. Create a simplified model performance plot (confusion matrix + ROC)
plt.figure(figsize=(12, 5))

# Example confusion matrix values
cm = np.array([[1200, 200], [100, 500]])
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'], 
            yticklabels=['Not Churned', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Example ROC curve
plt.subplot(1, 2, 2)
fpr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
tpr = np.array([0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 1.0])
roc_auc = 0.84
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig('static/images/project/churn/model_performance.png', dpi=300, bbox_inches='tight')
plt.close()

print("Model performance plot saved to static/images/project/churn/model_performance.png")

# 2. Create a simplified feature importance plot
plt.figure(figsize=(10, 8))
feature_importance = pd.DataFrame({
    'Feature': ['Tenure', 'Monthly Charges', 'Total Charges', 'Contract Type', 'Internet Service'],
    'Importance': [0.25, 0.20, 0.15, 0.12, 0.10]  
})
sns.barplot(x='Importance', y='Feature', data=feature_importance.sort_values('Importance', ascending=False))
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('static/images/project/churn/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

print("Feature importance plot saved to static/images/project/churn/shap_summary.png")

# 3. Create financial distributions plot
plt.figure(figsize=(12, 5))
# Generate sample financial data
monthly_charges = np.random.normal(65, 30, 1000)
total_charges = np.random.gamma(5, 200, 1000)

plt.subplot(1, 2, 1)
sns.histplot(monthly_charges, bins=30, kde=True)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charges ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(total_charges, bins=30, kde=True)
plt.title('Total Charges Distribution')
plt.xlabel('Total Charges ($)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('static/images/project/churn/financial_dist.png', dpi=300, bbox_inches='tight')
plt.close()

print("Financial distributions plot saved to static/images/project/churn/financial_dist.png")

# 4. Create service usage plot
plt.figure(figsize=(10, 6))
services = ['Internet Service', 'Phone Service', 'Tech Support', 'Online Backup', 'Device Protection']
churn_with_service = [25, 15, 18, 20, 22]
churn_without_service = [35, 40, 30, 28, 25]

x = np.arange(len(services))
width = 0.35

plt.bar(x - width/2, churn_with_service, width, label='With Service')
plt.bar(x + width/2, churn_without_service, width, label='Without Service')
plt.xlabel('Service Type')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Service Type')
plt.xticks(x, services, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig('static/images/project/churn/service_usage.png', dpi=300, bbox_inches='tight')
plt.close()

print("Service usage plot saved to static/images/project/churn/service_usage.png")

# 5. Create demographics plot
plt.figure(figsize=(10, 6))
demographics = ['Male', 'Female', 'Senior', 'Non-Senior', 'With Partner', 'Without Partner', 'With Dependents', 'Without Dependents']
churn_rates = [26, 28, 35, 22, 20, 32, 18, 30]

plt.bar(range(len(demographics)), churn_rates)
plt.xlabel('Demographic Group')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Demographics')
plt.xticks(range(len(demographics)), demographics, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('static/images/project/churn/demographics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Demographics plot saved to static/images/project/churn/demographics.png")

print("All plots have been generated successfully.") 