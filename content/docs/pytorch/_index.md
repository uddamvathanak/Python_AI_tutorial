---
title: "4. Introduction to Neural Networks with PyTorch"
weight: 40
---

Welcome to Module 4! We now transition from classical machine learning with scikit-learn to the foundations of **Deep Learning**. Neural Networks are powerful models inspired by the structure of the human brain, capable of learning complex patterns. **PyTorch** is a leading open-source deep learning framework known for its flexibility and Pythonic feel, widely used in both research and industry.

### Learning Objectives
After this module, you will be able to:
*   Understand the concept of a Perceptron and the role of activation functions.
*   Describe the structure of a basic Feedforward Neural Network.
*   Explain the core ideas behind Backpropagation and Gradient Descent for training neural networks.
*   Understand key PyTorch concepts: Tensors, Autograd, Modules (`nn.Module`), Optimizers, and Loss Functions.
*   Build and train a simple feedforward neural network in PyTorch for a basic task.
*   Grasp the fundamental idea behind Convolutional Neural Networks (CNNs) for image data.

{{< callout type="info" >}}
**Interactive Practice:**
Copy the code examples into your Jupyter Notebook environment ([Google Colab](https://colab.research.google.com/) or local). You'll need PyTorch installed. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions specific to your system (CPU/GPU). You can often install the CPU version easily via pip or conda: `pip install torch torchvision torchaudio` or `conda install pytorch torchvision torchaudio cpuonly -c pytorch`.
{{< /callout >}}

## Perceptrons and Activation Functions

The simplest unit in a neural network is often inspired by the **Perceptron**. Conceptually, it takes multiple inputs, computes a weighted sum, adds a bias, and then passes the result through an **activation function**.

*   **Weighted Sum + Bias:** \( z = (\sum_{i} w_i x_i) + b \) (where \(x_i\) are inputs, \(w_i\) are weights, \(b\) is bias)
*   **Activation Function:** Introduces non-linearity into the model, allowing it to learn complex relationships beyond simple linear combinations. Without non-linear activation functions, a deep neural network would just behave like a single linear layer.

**Common Activation Functions:**

*   **Sigmoid:** Squashes values between 0 and 1. \( \sigma(z) = \frac{1}{1 + e^{-z}} \). Often used in the output layer for binary classification.
*   **ReLU (Rectified Linear Unit):** \( \text{ReLU}(z) = \max(0, z) \). Very popular for hidden layers due to its simplicity and efficiency. It outputs the input directly if positive, otherwise, it outputs zero.
*   **Tanh (Hyperbolic Tangent):** Squashes values between -1 and 1. \( \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \).
*   **Softmax:** Used in the output layer for multi-class classification. Converts a vector of scores into a probability distribution (outputs sum to 1).

{{< callout type="tip" >}}
The choice of activation function is important. ReLU is a common default for hidden layers, while Sigmoid/Softmax are typical for output layers depending on the task (binary/multi-class classification).
{{< /callout >}}

## Feedforward Neural Networks

Also known as Multi-Layer Perceptrons (MLPs), these are the most basic type of artificial neural network.

*   **Structure:** Consists of an input layer, one or more hidden layers, and an output layer.
*   **Connections:** Neurons in one layer are typically fully connected to neurons in the next layer.
*   **Information Flow:** Data flows strictly in one direction – from the input layer, through the hidden layers, to the output layer – without loops (hence "feedforward").
*   **Learning:** The network "learns" by adjusting the weights and biases of the connections between neurons during training to minimize the difference between its predictions and the actual target values.

<img src="/images/pytorch/nn.svg" alt="Simple Feedforward Network Diagram (Conceptual)" style="width: 100%;">

## Backpropagation and Gradient Descent

How does the network learn to adjust its weights and biases?

1.  **Forward Pass:** Input data is fed through the network layer by layer, applying weighted sums and activation functions, until an output (prediction) is generated.
2.  **Loss Calculation:** A **loss function** (or cost function) measures how far the network's prediction is from the true target value. Common loss functions include Mean Squared Error (MSE) for regression and Cross-Entropy Loss for classification.
3.  **Backward Pass (Backpropagation):** This is the core learning algorithm. It calculates the **gradient** (derivative) of the loss function with respect to each weight and bias in the network. It uses the chain rule of calculus to efficiently propagate the error signal backward from the output layer to the input layer. The gradient indicates the direction and magnitude of change needed for each weight/bias to reduce the loss.
4.  **Weight Update (Gradient Descent):** An **optimizer** algorithm (like Stochastic Gradient Descent - SGD, or more advanced ones like Adam) uses the calculated gradients to update the weights and biases. It takes small steps in the opposite direction of the gradient to minimize the loss. The size of these steps is controlled by the **learning rate**.

This cycle (Forward Pass -> Loss Calculation -> Backward Pass -> Weight Update) is repeated many times (over many **epochs** and **batches** of data) until the model's performance converges.

{{< callout type="info" >}}
You don't usually implement backpropagation manually. Deep learning frameworks like PyTorch automatically calculate the gradients using a system called **Autograd**. You define the network structure and the loss function, and the framework handles the gradient computation.
{{< /callout >}}

## Building and Training Models in PyTorch

Let's see how these concepts translate into PyTorch code.

**Core PyTorch Concepts:**

*   **Tensors:** The fundamental data structure in PyTorch, similar to NumPy arrays but with added capabilities for GPU acceleration and automatic differentiation.
*   **`torch.nn.Module`:** The base class for all neural network modules (layers, or the entire network itself). You define your network architecture by subclassing `nn.Module`.
*   **`torch.autograd`:** PyTorch's automatic differentiation engine. It tracks operations on tensors and automatically computes gradients during the backward pass.
*   **Loss Functions (`torch.nn`):** Pre-defined loss functions (e.g., `nn.MSELoss`, `nn.CrossEntropyLoss`).
*   **Optimizers (`torch.optim`):** Implementations of optimization algorithms (e.g., `optim.SGD`, `optim.Adam`) used to update model weights.

**Simple Feedforward Network Example (Binary Classification):**

```python
import torch
import torch.nn as nn # Neural network modules
import torch.optim as optim # Optimization algorithms
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification # Generate synthetic data
from sklearn.preprocessing import StandardScaler

# 1. Generate Synthetic Data (using scikit-learn for convenience)
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

# Convert to PyTorch Tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Target needs shape (n_samples, 1) for BCELoss

# Scale features (important for neural networks)
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the Network Architecture
class SimpleClassifier(nn.Module):
    def __init__(self, num_features):
        super(SimpleClassifier, self).__init__() # Call parent class constructor
        self.layer_1 = nn.Linear(num_features, 16) # Input features -> 16 hidden units
        self.activation_1 = nn.ReLU()             # ReLU activation
        self.layer_2 = nn.Linear(16, 8)           # 16 hidden units -> 8 hidden units
        self.activation_2 = nn.ReLU()
        self.output_layer = nn.Linear(8, 1)       # 8 hidden units -> 1 output unit (for binary)
        self.output_activation = nn.Sigmoid()     # Sigmoid for binary probability

    # Define the forward pass
    def forward(self, x):
        x = self.layer_1(x)
        x = self.activation_1(x)
        x = self.layer_2(x)
        x = self.activation_2(x)
        x = self.output_layer(x)
        x = self.output_activation(x)
        return x

# Create an instance of the model
input_features = X_train.shape[1]
model = SimpleClassifier(num_features=input_features)
print("Model Architecture:\n", model)

# 3. Define Loss Function and Optimizer
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam optimizer

# 4. Training Loop
num_epochs = 100
batch_size = 32 # Process data in batches

for epoch in range(num_epochs):
    model.train() # Set model to training mode

    # Simple batching (usually use DataLoader for efficiency)
    permutation = torch.randperm(X_train.size()[0])
    
    for i in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad() # Clear previous gradients

        indices = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimize
        loss.backward() # Calculate gradients (Autograd magic!)
        optimizer.step() # Update weights

    # Print loss every few epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 5. Evaluation
model.eval() # Set model to evaluation mode (disables dropout, etc.)
with torch.no_grad(): # Disable gradient calculation for evaluation
    y_pred_tensor = model(X_test)
    y_pred_binary = (y_pred_tensor >= 0.5).float() # Convert probabilities to 0 or 1

    accuracy = (y_pred_binary == y_test).sum().item() / y_test.shape[0]
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

```

## Convolutional Neural Networks (CNNs) Basics

While feedforward networks work well for tabular data, they aren't ideal for data with spatial structure, like images. **Convolutional Neural Networks (CNNs)** are a specialized type of neural network designed primarily for processing grid-like data (e.g., images).

**Key Concepts:**

*   **Convolutional Layers:** Instead of fully connected layers, CNNs use convolutional layers that apply learnable filters (kernels) across the input image. These filters detect spatial patterns like edges, corners, textures, etc.
*   **Pooling Layers:** Reduce the spatial dimensions (width/height) of the feature maps, making the model more robust to variations in the position of features and reducing computational load. Max Pooling is common.
*   **Feature Hierarchy:** Early layers learn simple features (edges), while deeper layers combine these to learn more complex features (shapes, objects).

CNNs have revolutionized computer vision tasks like image classification, object detection, and segmentation.

{{< callout type="tip" >}}
We'll explore CNNs in more detail in the Computer Vision module. PyTorch provides convenient layers like `nn.Conv2d` and `nn.MaxPool2d` to build CNN architectures.
{{< /callout >}}

## Practice Exercises (Take-Home Style)

1.  **Activation Functions:** Briefly describe why non-linear activation functions (like ReLU or Sigmoid) are necessary in multi-layer neural networks. What would happen if you only used linear activation functions?
    *   _Expected Result:_ Non-linearities allow the network to learn complex, non-linear relationships in the data. Without them, a multi-layer network would mathematically collapse into a single linear transformation, unable to model complex patterns.
2.  **PyTorch Tensors:** Create a 2x3 PyTorch tensor filled with random numbers. Print the tensor and its shape.
    *   _Expected Result:_ Output will show a 2x3 tensor with random values and its shape `torch.Size([2, 3])`.
3.  **Define a Simple Network:** Define a PyTorch `nn.Module` class for a network with one hidden layer containing 8 neurons and using the ReLU activation function. Assume the input has 5 features and the output predicts a single continuous value (regression - no output activation needed here). Don't worry about training it.
    *   _Expected Result:_ A class definition inheriting from `nn.Module` with an `__init__` method defining `nn.Linear(5, 8)` and `nn.Linear(8, 1)` layers, and a `forward` method applying the layers sequentially with `nn.ReLU()` after the first linear layer.
4.  **Loss Function Choice:** Which PyTorch loss function (`nn.MSELoss` or `nn.BCELoss`) would be appropriate if you were training the network from Exercise 3 for a regression task? Why?
    *   _Expected Result:_ `nn.MSELoss` (Mean Squared Error Loss) is appropriate for regression tasks where the goal is to minimize the squared difference between continuous predicted and actual values. `nn.BCELoss` is for binary classification.

## Summary

You've been introduced to the fundamental building blocks of neural networks: perceptrons, activation functions, and the feedforward architecture. We discussed the crucial training process involving loss functions, backpropagation (gradient calculation), and gradient descent (weight updates via optimizers). You saw how to implement and train a basic neural network using PyTorch's core components (`Tensor`, `nn.Module`, `optim`, loss functions, `autograd`). Finally, we briefly touched upon CNNs, specialized networks for image data.

## Additional Resources

*   **[PyTorch Official Tutorials](https://pytorch.org/tutorials/):** Excellent starting point, covering basics to advanced topics.
*   **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html):** Comprehensive API reference.
*   **[Deep Learning with PyTorch Book (Online)](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf):** A thorough book available freely online.
*   **[3Blue1Brown - Neural Networks (YouTube Playlist)](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi):** Fantastic visual intuition for how neural networks work and learn.
*   **[fast.ai Course](https://course.fast.ai/):** Practical deep learning course using PyTorch (and their own library built on top).

**Next:** Let's shift focus to processing language data. Proceed to [Module 5: Natural Language Processing (NLP) Essentials](/docs/nlp/). 