---
title: "1. Python Fundamentals for AI"
weight: 10
---

Welcome to the first module! Before diving into complex AI libraries, it's crucial to have a solid grasp of Python's fundamentals. Python's clear syntax and vast ecosystem make it the primary language for AI, Machine Learning, and Data Science. This section covers the essential building blocks you'll use constantly.

### Learning Objectives
After this module, you will be able to:
*   Declare variables and identify common Python data types (`int`, `float`, `str`, `bool`, `list`, `dict`).
*   Control program flow using `if`/`elif`/`else` statements and `for`/`while` loops.
*   Define and call your own functions.
*   Import and use modules and libraries.
*   Understand the basic concepts of classes and objects.
*   Perform basic file reading and writing operations.

{{< callout type="info" >}}
**Interactive Practice:**
You can copy the code examples from this page and run them in your own Jupyter Notebook environment (either running locally or using a cloud service like [Google Colab](https://colab.research.google.com/)). This is a great way to experiment and solidify your understanding!
{{< /callout >}}

## Variables and Data Types

Variables are used to store information that your program can manipulate. In Python, you assign a value to a variable name using the equals sign (`=`). Python automatically determines the data type based on the value assigned.

**Common Data Types:**

*   **`int` (Integer):** Whole numbers (e.g., `10`, `-5`, `0`).
*   **`float` (Floating-Point):** Numbers with decimals (e.g., `3.14`, `-0.5`).
*   **`str` (String):** Sequences of characters (text) enclosed in quotes (e.g., `"Hello"`, `'AI'`).
*   **`bool` (Boolean):** Represents truth values, either `True` or `False`.
*   **`list`:** Ordered, mutable (changeable) sequence of items. Enclosed in `[]`. (e.g., `[1, 2, 3]`, `['apple', 0.5]`)
*   **`dict` (Dictionary):** Unordered collection of key-value pairs. Enclosed in `{}`. (e.g., `{'learning_rate': 0.01, 'epochs': 10}`)

```python
# Example variable assignments
learning_rate = 0.001     # float
epochs = 50               # int
model_name = "Classifier" # str
is_training = True        # bool
feature_list = [1.2, 3.4, 0.9] # list
hyperparameters = {       # dict
    "batch_size": 32,
    "optimizer": "Adam"
}

print(f"Model: {model_name}, Learning Rate: {learning_rate}")
print(f"Features: {feature_list}")
# (Note: We use f-strings (formatted string literals, like f"...") for easily embedding variable values within strings.)
```

{{< callout type="tip" >}}
**Why this matters for AI:** You'll use variables constantly to store configuration (like `learning_rate`), track progress (`epochs`, `is_training`), hold your data (`feature_list`), and define model parameters (`hyperparameters`). Understanding the difference between types like lists (ordered sequences) and dictionaries (key-value lookup) is crucial for organizing data.
{{< /callout >}}

## Control Structures (if/else, loops)

Control structures allow you to dictate the flow of your program's execution.

**Conditional Statements (`if`, `elif`, `else`)**

These structures execute different blocks of code depending on whether a condition is `True` or `False`.

```python
validation_accuracy = 0.85

if validation_accuracy > 0.9:
    print("Excellent model performance!")
elif validation_accuracy > 0.7:
    print("Good model performance, consider further tuning.")
else:
    print("Model needs significant improvement.")
```

**Loops (`for`, `while`)**

Loops are used to repeat a block of code multiple times.

*   **`for` loop:** Iterates over a sequence (like a list) or a range of numbers.
*   **`while` loop:** Repeats as long as a condition remains `True`.

```python
# For loop iterating over a list of layers
layers = ['input', 'hidden1', 'hidden2', 'output']
for layer in layers:
    print(f"Processing layer: {layer}")

# For loop iterating a specific number of times (e.g., training epochs)
num_epochs = 10
for epoch in range(num_epochs): # range(10) generates numbers 0 through 9
    print(f"Starting Epoch {epoch + 1}")
    # --- Training code for one epoch would go here ---
    pass # pass is a placeholder that does nothing

# While loop example (less common for basic iteration)
count = 0
while count < 3:
    print(f"Iteration {count}")
    count += 1 # Increment count (important to avoid infinite loop)
```

{{< callout type="info" >}}
**AI Relevance:** Conditionals are used everywhere, from checking data validity to deciding model behavior based on performance metrics. `for` loops are fundamental for iterating through datasets (batches of images, text samples), running training epochs, and processing features.
{{< /callout >}}

## Functions and Modules

**Functions**

Functions are named blocks of reusable code designed to perform a specific task. They help make your code organized, readable, and easier to debug. You define a function using the `def` keyword.

```python
def preprocess_data(raw_data):
  """Cleans and prepares the input data.""" # Docstring explaining the function
  print(f"Preprocessing {len(raw_data)} data points...")
  # --- Actual preprocessing steps would go here ---
  processed = [item * 10 for item in raw_data] # Example step (using a List Comprehension)
  # List comprehensions provide a concise way to create lists.
  return processed

data = [1, 2, 3, 4]
cleaned_data = preprocess_data(data)
print(f"Processed data: {cleaned_data}")
```

**Modules**

Modules are Python files (`.py`) containing definitions and statements. They allow you to import functionality written by others or yourself. The core AI ecosystem relies heavily on importing modules (libraries).

```python
# Import the entire 'math' module
import math
print(math.sqrt(25)) # Output: 5.0

# Import a specific function 'sqrt' from 'math'
from math import sqrt
print(sqrt(25)) # Output: 5.0

# Import a library with a common alias (you'll see this a LOT)
import numpy as np # NumPy is fundamental for numerical operations
import pandas as pd # Pandas is key for data manipulation

# Now you can use functions from these libraries via their alias
# Example (we'll cover NumPy/Pandas in detail later):
# data_array = np.array([1, 2, 3])
# print(data_array)
```

{{< callout type="tip" >}}
Defining functions like `preprocess_data`, `train_model`, `evaluate_performance` helps structure your AI projects logically. Mastering imports is essential, as you'll constantly use libraries like `numpy`, `pandas`, `sklearn`, `pytorch`, etc.
{{< /callout >}}

## Object-Oriented Programming (OOP) Basics

OOP is a way of structuring programs around "objects" that combine data (attributes) and functions that operate on that data (methods). A `class` serves as a blueprint for creating objects.

While you can achieve much without deep OOP, understanding the basics is helpful as many AI libraries are object-oriented.

```python
class BasicDatasetLoader:
    # Constructor: Initializes the object when created
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None # Attribute to store data later

    # Method: A function belonging to the class
    def load_data(self):
        print(f"Loading data from {self.file_path}...")
        # --- In reality, complex file reading logic here ---
        self.data = ["sample1", "sample2", "sample3"] # Example data
        print("Data loaded.")

    # Method: Another function operating on the object's data
    def get_batch(self, batch_size):
        if self.data:
            print(f"Getting batch of size {batch_size}")
            return self.data[:batch_size] # Return first 'batch_size' items
        else:
            print("Data not loaded yet.")
            return []

# Create an object (instance) of the class
my_loader = BasicDatasetLoader("path/to/my/data.csv")

# Call methods on the object
my_loader.load_data()
batch = my_loader.get_batch(2)
print(f"Received batch: {batch}")
```

{{< callout type="info" >}}
You'll encounter classes when using frameworks like PyTorch or TensorFlow to define model architectures, or when using Scikit-learn's estimator objects (`model = RandomForestClassifier()`). This example shows how a class can encapsulate data loading logic.
{{< /callout >}}

## File Handling and Basic I/O

AI often involves reading data from files or saving results/models. Python provides built-in functions for basic file input/output (I/O).

The `with open(...) as ...:` syntax is the standard, safe way to work with files.

```python
# Example: Writing model performance logs
try: # The try block lets you test a block of code for errors.
    with open("training_log.txt", "w") as log_file: # 'w' = write (overwrites)
        log_file.write("Epoch 1: Accuracy=0.85\n")
        log_file.write("Epoch 2: Accuracy=0.88\n")
    print("Log file written successfully.")

    with open("training_log.txt", "a") as log_file: # 'a' = append (adds to end)
        log_file.write("Epoch 3: Accuracy=0.91\n")
    print("Appended to log file.")

    # Example: Reading the log file
    with open("training_log.txt", "r") as log_file: # 'r' = read (default)
        print("\nReading log file content:")
        log_content = log_file.read()
        print(log_content)

except IOError as e: # The except block handles the error if one occurs in the try block.
    print(f"An error occurred during file operation: {e}")

```

{{< callout type="warning" >}}
Always use `with open(...)`! It ensures the file is properly closed automatically, even if errors occur during reading or writing. For structured data like CSV or JSON, libraries like Pandas offer much more powerful and convenient loading functions, which we'll see later.
{{< /callout >}}

## Practice Exercises (Take-Home Style)

Try these exercises on your own to reinforce the concepts:

1.  **Model Configuration:** Create a Python dictionary named `model_config` to store the following hyperparameters: `learning_rate` (set to `0.05`), `optimizer` (set to `'SGD'`), `epochs` (set to `100`), and `layers` (a list containing the strings `'input'`, `'dense_128'`, `'output'`). Print the dictionary.
    *   _Expected Result:_ The printed output should look similar to this (order of keys might vary):
        ```
        {'learning_rate': 0.05, 'optimizer': 'SGD', 'epochs': 100, 'layers': ['input', 'dense_128', 'output']}
        ```

2.  **Accuracy Check:** Write an `if/else` statement that checks if a variable `test_accuracy` (assign it a value, e.g., `0.65`) is greater than `0.75`. Print "Model passed!" if it is, and "Model failed." otherwise.
    *   _Expected Result (for `test_accuracy = 0.65`):_ `Model failed.`
    *   _Expected Result (if you try `test_accuracy = 0.80`):_ `Model passed!`

3.  **Simple Function:** Define a function called `add_two_numbers` that takes two arguments (`num1`, `num2`) and returns their sum. Call the function with `5` and `10` and print the result.
    *   _Expected Result:_ `15`

4.  **Looping:** Write a `for` loop that iterates through the `layers` list you created in Exercise 1 and prints each layer name.
    *   _Expected Result:_ The output should print each layer name on a new line:
        ```
        input
        dense_128
        output
        ```

5.  **File Writing:** Use the `with open(...)` syntax to create a new file named `results.txt` and write the string "Training complete." into it.
    *   _Expected Result:_ No direct output printed to the console, but a new file named `results.txt` should be created in the same directory where you run the script, containing the text "Training complete.".

*(Feel free to experiment and modify these exercises!)*

## Summary

You've covered the essential Python syntax and structures: variables & types, control flow, functions & modules, basic OOP concepts, and file I/O. These are the tools you'll use in almost every AI script or notebook.

## Additional Resources

For further learning or different perspectives on these fundamental concepts, check out these excellent resources:

*   **[The Official Python Tutorial](https://docs.python.org/3/tutorial/index.html):** The definitive source, covering everything in great detail.
*   **[Codecademy - Learn Python 3](https://www.codecademy.com/learn/learn-python-3):** Offers interactive exercises for learning Python syntax.
*   **[freeCodeCamp - Scientific Computing with Python Certification](https://www.freecodecamp.org/learn/scientific-computing-with-python/):** Includes a comprehensive Python curriculum often used for data science foundations.

**Next:** Ready to handle data efficiently? Proceed to [Module 2: Data Wrangling & Analysis: NumPy & Pandas](/Python_AI_tutorial/docs/numpy-pandas/). 