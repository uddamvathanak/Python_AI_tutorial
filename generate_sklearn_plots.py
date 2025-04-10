import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Generating plots for Module 3...")

# --- Ensure output directory exists ---
output_dir = 'static/images/sklearn/'
os.makedirs(output_dir, exist_ok=True)
print(f"Ensured directory exists: {output_dir}")

# === Confusion Matrix Plot ===
print("Generating Confusion Matrix plot...")
try:
    # --- Create Dummy Data for Classification ---
    # Generate some separable data for a clearer plot
    np.random.seed(42) # for reproducibility
    X_c1 = np.random.rand(50, 2) + [0, 0] # Class 0
    X_c2 = np.random.rand(50, 2) + [1, 1] # Class 1
    X_clf = np.vstack((X_c1, X_c2))
    y_clf = np.array([0]*50 + [1]*50)

    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42, stratify=y_clf
    )

    # Train a simple model
    model = LogisticRegression()
    model.fit(X_train_clf, y_train_clf)
    predictions = model.predict(X_test_clf)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test_clf, predictions)

    # Plot using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5)) # Create figure and axes explicitly
    disp.plot(ax=ax, cmap=plt.cm.Blues) # Plot on the axes
    ax.set_title('Confusion Matrix Example') # Set title on axes

    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig) # Close the specific figure
    print(f"Saved plot to {save_path}")

    print("\nPlot generated successfully!")

except Exception as e:
    print(f"\nAn error occurred during plot generation: {e}") 