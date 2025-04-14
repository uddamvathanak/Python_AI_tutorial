import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Create directory if needed
os.makedirs('static/images/project/churn', exist_ok=True)

# 1. Create an improved demographics plot with better spacing and sizing
plt.figure(figsize=(12, 8))  # Larger figure size
demographics = ['Male', 'Female', 'Senior', 'Non-Senior', 'With Partner', 'Without Partner', 'With Dependents', 'Without Dependents']
churn_rates = [26, 28, 35, 22, 20, 32, 18, 30]

# Use a more visually appealing color palette
colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#34495E']
bars = plt.bar(range(len(demographics)), churn_rates, color=colors)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Demographic Group', fontsize=12)
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.title('Customer Churn Rate by Demographics', fontsize=14, fontweight='bold')
plt.xticks(range(len(demographics)), demographics, rotation=45, ha='right', fontsize=11)
plt.ylim(0, max(churn_rates) + 8)  # Add some space at the top for labels
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add tight layout with more bottom padding for rotated labels
plt.tight_layout(pad=2.0, rect=[0, 0.05, 1, 0.95])
plt.savefig('static/images/project/churn/demographics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Improved demographics plot saved")

# 2. Create an improved service usage plot with better spacing
plt.figure(figsize=(14, 8))  # Wider figure for more space
services = ['Internet Service', 'Phone Service', 'Tech Support', 'Online Backup', 'Device Protection']
churn_with_service = [25, 15, 18, 20, 22]
churn_without_service = [35, 40, 30, 28, 25]

x = np.arange(len(services))
width = 0.35

# Use more contrasting colors
plt.bar(x - width/2, churn_with_service, width, label='With Service', color='#3498DB')
plt.bar(x + width/2, churn_without_service, width, label='Without Service', color='#E74C3C')

# Add data labels on top of each bar
for i, v in enumerate(churn_with_service):
    plt.text(i - width/2, v + 0.5, f'{v}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
for i, v in enumerate(churn_without_service):
    plt.text(i + width/2, v + 0.5, f'{v}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.xlabel('Service Type', fontsize=12)
plt.ylabel('Churn Rate (%)', fontsize=12)
plt.title('Churn Rate Comparison by Service Type', fontsize=14, fontweight='bold')
plt.xticks(x, services, fontsize=11)
plt.ylim(0, max(churn_without_service) + 8)  # Add space for labels
plt.legend(fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add annotation explaining the significance
plt.annotate('Customers without technical services show\nsignificantly higher churn rates',
             xy=(2.5, 10), xytext=(2.5, 5),
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
             ha='center', fontsize=10)

plt.tight_layout(pad=2.0)
plt.savefig('static/images/project/churn/service_usage.png', dpi=300, bbox_inches='tight')
plt.close()

print("Improved service usage plot saved")

print("Plot generation completed successfully!") 