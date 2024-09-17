# Linear Feature  Engineering

## Overview
This project is a requirement for our coursework CISC 820:  Quantitative Foundations for the first year of PhD in Computing and Information Sciences at RIT. The core logic is implemented in the `main.py` script, and all configurations are managed through the `config.yaml` file. The results, including predictions and output figures, are saved in the `artifacts` folder. The final predicted results from the given test inputs are in [test_prediction.txt](artifacts/test_prediction.txt).

## Table of Contents
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Authors](#authors)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SrjPdl/Linear-Feature-Engineering.git
    cd Linear-Feature-Engineering
    ```

2. **Create a virtual environment**:
    You should create a virtual environment to isolate the project's dependencies.
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On **Windows**:
        ```bash
        venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies**:
    All the required packages are listed in `requirements.txt`. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All configurations for the project are stored in the `config.yaml` file. The configuration includes:
- **k_folds**: Specifies the number of folds to use for cross-validation during model training.
- **max_degree**: Defines the maximum polynomial degree to be used for model fitting.
- **train_file_path**: Path to the file containing the training dataset.
- **test_file_path**: Path to the file containing the test dataset.
- **prediction_file_path**: Path where the predictions will be saved after the model runs.

# Running the project
To run the project, simply execute the main.py script. Make sure you have activated your virtual environment before running the script.
 ```bash
    python main.py
```

# Authors
- Suraj Poudel
- Image Adhikari

# License
This project is licensed under the [MIT](https://www.mit.edu/~amini/LICENSE.md) License. 

