import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Generating plots for Module 2...")

# --- Ensure output directory exists ---
output_dir = 'static/images/numpy-pandas/'
# Create the directory structure if it doesn't exist.
# os.makedirs will create parent directories as needed and won't raise an error if the directory already exists.
os.makedirs(output_dir, exist_ok=True) 
print(f"Ensured directory exists: {output_dir}")

# --- Create Dummy Data ---
try:
    print("Creating dummy data.csv...")
    # Create a dummy CSV file in the current directory for demonstration purposes.
    with open('data.csv', 'w') as f:
        f.write("ID,Temperature,Humidity\n")
        f.write("1,25.5,60\n")
        f.write("2,26.1,62\n")
        f.write("3,24.9,58\n")
        f.write("4,,55\n") # Intentionally add a row with missing temperature data.
    
    # Read the dummy CSV file into a Pandas DataFrame.
    df_from_csv = pd.read_csv('data.csv')
    print("Loaded data.csv into DataFrame.")

    # --- Handle Missing Values ---
    # Calculate the mean of the 'Temperature' column, ignoring NaN values.
    mean_temp = df_from_csv['Temperature'].mean() 
    # Fill missing values (NaN) in the 'Temperature' column with the calculated mean.
    # fillna returns a new DataFrame with missing values filled.
    df_filled = df_from_csv.fillna({'Temperature': mean_temp}) 
    # Use this filled DataFrame for subsequent plotting.
    df_plot = df_filled 
    print("Filled missing temperature values.")

    # === Plot 1: Temperature Trend (Matplotlib) ===
    print("Generating Temperature Trend plot...")
    plt.figure(figsize=(8, 4)) # Create a new figure for the plot with a specified size.
    # Plot 'ID' on the x-axis and 'Temperature' on the y-axis using lines and markers.
    plt.plot(df_plot['ID'], df_plot['Temperature'], marker='o', linestyle='-') 
    plt.title('Temperature Trend') # Set the title of the plot.
    plt.xlabel('ID') # Set the label for the x-axis.
    plt.ylabel('Temperature (°C)') # Set the label for the y-axis.
    plt.grid(True) # Add a grid to the plot for better readability.
    # Construct the full path for saving the plot image.
    save_path = os.path.join(output_dir, 'temp_trend.png') 
    # Save the current figure to the specified path. bbox_inches='tight' adjusts plot to prevent labels being cut off.
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close() # Close the figure to free up memory.
    print(f"Saved plot to {save_path}")

    # === Plot 2: Humidity Distribution (Seaborn) ===
    print("Generating Humidity Distribution plot...")
    plt.figure(figsize=(8, 4))
    # Create a histogram of the 'Humidity' column using Seaborn. kde=True adds a Kernel Density Estimate curve.
    sns.histplot(data=df_plot, x='Humidity', kde=True) 
    plt.title('Humidity Distribution')
    save_path = os.path.join(output_dir, 'humidity_hist.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

    # === Plot 3: Temperature vs. Humidity (Seaborn) ===
    print("Generating Temperature vs. Humidity scatter plot...")
    plt.figure(figsize=(8, 4))
    # Create a scatter plot showing the relationship between 'Temperature' and 'Humidity'.
    sns.scatterplot(data=df_plot, x='Temperature', y='Humidity') 
    plt.title('Temperature vs. Humidity')
    save_path = os.path.join(output_dir, 'temp_vs_humidity_scatter.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

    # === Plot 4: Temperature Distribution by Category (Seaborn) ===
    print("Generating Temperature by Category boxplot...")
    # Add a dummy 'Category' column to the DataFrame for demonstration purposes.
    df_plot['Category'] = ['A', 'A', 'B', 'B'] 
    plt.figure(figsize=(8, 4))
    # Create a box plot comparing 'Temperature' distributions across different 'Category' values.
    sns.boxplot(data=df_plot, x='Category', y='Temperature') 
    plt.title('Temperature Distribution by Category')
    save_path = os.path.join(output_dir, 'temp_by_cat_boxplot.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

    print("\nAll plots generated successfully!")

# Handle the case where the dummy CSV file might not be found (though unlikely as we create it).
except FileNotFoundError: 
    print("\nError: data.csv could not be created or read.")
# Catch any other unexpected errors during the process.
except Exception as e: 
    print(f"\nAn error occurred: {e}")
# The finally block ensures that the dummy CSV file is deleted regardless of whether errors occurred.
finally: 
    if os.path.exists('data.csv'):
        os.remove('data.csv')
        print("Cleaned up dummy data.csv file.") 