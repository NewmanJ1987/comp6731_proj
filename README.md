# COMP 6731 Project: Cross-Entropy vs. Dense Max-Margin Loss

This project compares the performance of Cross-Entropy (CE) loss with Dense Max-Margin Loss (DMML) on two classification datasets: **Dermatology** and **Healthcare (Heart Disease)**.

## Project Structure

```
.
├── __init__.py
├── dermatology
│   ├── ce_vs_dmml.py            # Dermatology ce vs dmml comparison
│   ├── dermatology.csv          # Dermatology datasets
│   └── trial.ipynb              # Script measuring accuracy of dermatology
├── healthcare
│   ├── __init__.py
│   ├── ce_vs_dmml.py           # Healthcare ce vs dmml comparison
│   ├── heart.csv               # Heart disease datasets 
│   └── implementation.py       # Script measuring accuracy of healthcare
├── README.md
├── requirements.txt            # The dependencies for this project
└── training_utilitites.py      # Shared training utilities
```

## Installation & Setup

### 1. Create and Activate Virtual Environment

```bash
# Navigate to project directory
cd ~/workspace/comp_6731/project

# Create virtual environment (if not already created)
python3 -m venv env_project

# Activate virtual environment
source env_project/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

The project uses a `.env` file to store dataset paths. This makes the code portable and allows you to run experiments from any location without modifying the source code.

#### Understanding the `.env` File

A `.env` file is a simple text file that stores environment variable key-value pairs. The project includes a `.env.example` file at the root directory with the following structure:

```bash
# Dataset paths
HEART_DATASET_CSV=/path/to/heart.csv
DERMATOLOGY_DATASET_CSV=/path/to/dermatology.csv
```

#### How to Configure `.env`


1. **If the `.env` file doesn't exist:**
   - Create it manually:
     ```bash
     cp .env.example .env
     ```
   - Update the template with paths to match your system:
     ```bash
     # Dataset paths
     HEART_DATASET_CSV=/path/to/your/heart.csv
     DERMATOLOGY_DATASET_CSV=/path/to/your/dermatology.csv
     ```


#### Example `.env` File Configurations

**macOS/Linux Example:**
```bash
HEART_DATASET_CSV=/home/user/workspace/comp_6731/project/healthcare/heart.csv
DERMATOLOGY_DATASET_CSV=/home/user/workspace/comp_6731/project/dermatology/dermatology.csv
```

**Windows Example:**
```bash
HEART_DATASET_CSV=C:\Users\user\workspace\comp_6731\project\healthcare\heart.csv
DERMATOLOGY_DATASET_CSV=C:\Users\user\workspace\comp_6731\project\dermatology\dermatology.csv
```

## Running the Experiments

### Run Dermatology Experiment

From the project root directory:

```bash
# Activate virtual environment first
source env_project/bin/activate

# Run dermatology CE vs DMML comparison

python -m dermatology.ce_vs_dmml
```

**Output:**
- Console logs showing training progress (CE, DMML-Gaussian)
- Training loss plot
- Validation accuracy plot
- PCA visualization of raw features
- t-SNE embeddings for DMML-G and CE models

### Run Healthcare Experiment

From the project root directory:

```bash
# Activate virtual environment first
source env_project/bin/activate

# Run healthcare CE vs DMML comparison
python healthcare.ce_vs_dmml
```

**Output:**
- Console logs showing training progress
- Training loss and accuracy curves
- t-SNE embeddings visualization
- Performance metrics comparison



## Key Features
### Loss Functions Compared

1. **Cross-Entropy (CE)**: Standard classification loss
3. **DMML-Gaussian**: Margin-based loss using Gaussian similarities

### Dataset Details

#### Dermatology
- **Features**: 34 (after removing missing values)
- **Classes**: 6 dermatology conditions
- **Samples**: ~366 total
- **Preprocessing**: Missing values ('?') filled with column medians, features standardized

#### Healthcare (Heart Disease)
- **Features**: ~14 (after categorical encoding)
- **Classes**: 2 (presence/absence of heart disease)
- **Preprocessing**: Categorical variables one-hot encoded, features standardized

## Configuration

### Model Architecture

All experiments use a 2-layer MLP:
```
Input Features → Linear(input_dim, 64) → ReLU → Linear(64, 64) → ReLU → Classifier
```

### Training Parameters

- **Optimizer**: Adam (learning rate 1e-3)
- **Epochs**: 100 (with early stoppage.)
- **Batch Size**: 32 (training), 128 (validation)
- **Train/Val Split**: 80/20 

### DMML Hyperparameters

- **Margin**: 1.0 (distance margin)
- **Beta**: 2.0 (similarity margin, Gaussian variant)
- **Sigma**: 1.0 (Gaussian bandwidth)
- **Weights**: ce_weight=1.0, mm_weight=1.0, var_weight=0.1

## Important Notes

### Device Selection

The code automatically uses GPU (CUDA) if available, otherwise falls back to CPU. Device info is printed at startup:
```
Using device: cuda
# or
Using device: cpu
```


## Output Interpretation

### Training Logs

```
[CE] Epoch 01  Loss=0.6234  Acc=0.7123
[DMML-S] Epoch 01  Loss=0.5821  Acc=0.7456
[DMML-G] Epoch 01  Loss=0.5634  Acc=0.7589
```

- **Loss**: Average loss per sample (lower is better)
- **Acc**: Validation accuracy (higher is better)

### Visualizations

1. **Training Loss Curve**: Compares convergence speed of the two methods
2. **Validation Accuracy Curve**: Shows final performance
3. **PCA Plot**: 2D projection of raw input features colored by class
4. **t-SNE Embedding**: 2D projection of learned representations, shows cluster separation

Better separated clusters in t-SNE indicate stronger learned representations.

## Virtual Environment Info

The `env_project` environment uses Python 3.x with the following key packages:

```
torch==2.2.0
scikit-learn==1.7.2
pandas==2.2.1
numpy==1.26.4
matplotlib==3.8.3
pytest==9.0.1
```

Run `pip freeze > requirements.txt` to update the requirements file if you add/remove packages.

## License & Credits

Part of COMP 6731 coursework.
