---
title: "Getting Started: Setting Up Your Environment"
weight: 1 # Makes this the first item in the 'docs' section sidebar
---

Welcome to the Python Essentials for AI Applications tutorial! To follow along with the code examples and complete the exercises, you'll need a Python environment with specific libraries installed.

We recommend using **Miniconda** to manage your Python environments and packages. Jupyter Notebooks will be our primary tool for interactive coding.

Choose the setup method that works best for you:

## Option 1: Local Setup with Miniconda (Recommended)

This gives you a dedicated environment on your own computer.

1.  **Install Miniconda:**
    *   Download the Miniconda installer for your operating system (Windows, macOS, Linux) from the official Conda documentation: [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/)
    *   Follow the installation instructions provided on the download page. Accept the default settings unless you have specific reasons to change them.

2.  **Create a Conda Environment:**
    *   Open your terminal (Anaconda Prompt on Windows, Terminal on macOS/Linux).
    *   Create a new environment specifically for this tutorial. We'll name it `pyai_env` and install Python 3.10 (or a recent version):
        ```bash
        conda create --name pyai_env python=3.10 -y
        ```
    *   Activate the new environment:
        ```bash
        conda activate pyai_env
        ```
        You should see `(pyai_env)` appear at the beginning of your terminal prompt.

3.  **Install Jupyter Notebook:**
    *   With the `pyai_env` environment active, install Jupyter:
        ```bash
        conda install jupyter -y
        ```

4.  **Install Core Libraries (Optional - can be done later):**
    *   You can pre-install the main libraries we'll use now, or install them as needed within the tutorial notebooks:
        ```bash
        conda install numpy pandas scikit-learn matplotlib seaborn pytorch torchvision torchaudio cpuonly -c pytorch -y
        conda install opencv-python -y 
        # For NLP libraries, we might install others later (e.g., nltk, transformers)
        ```
       *Note:* The PyTorch command installs the CPU-only version. If you have a compatible NVIDIA GPU and want GPU acceleration, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) for the correct CUDA-enabled command.

5.  **Launch Jupyter Notebook:**
    *   Make sure your `pyai_env` is active in the terminal.
    *   Navigate to the directory where you want to save your tutorial notebooks (or where you cloned the tutorial repository if applicable).
    *   Run the command:
        ```bash
        jupyter notebook
        ```
    *   This will open the Jupyter Notebook interface in your web browser. You can now create new notebooks or open existing `.ipynb` files.

## Option 2: Google Colaboratory (Colab)

If you prefer not to install software locally or want a cloud-based solution, Google Colab provides a free Jupyter Notebook environment with many common AI libraries pre-installed.

*   **Access Colab:** Go to [https://colab.research.google.com/](https://colab.research.google.com/)
*   **Usage:** You can create new notebooks or upload notebooks from your computer or GitHub.
*   **Installation:** While many libraries are pre-installed, you might occasionally need to install specific packages within a Colab notebook cell using `!pip install package-name`.
*   **Limitations:** Be aware of usage limits (GPU/TPU time, RAM) on the free tier. Your environment is also not persistent; you'll need to re-run setup cells or reinstall packages if your session disconnects.

---

With your environment set up, you're ready to dive into the first module! Proceed to the next section in the documentation. 