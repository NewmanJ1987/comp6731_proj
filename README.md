# COMP 6731 Project: Cross-Entropy vs. Dense Max-Margin Loss

This project compares the performance of Cross-Entropy (CE) loss with Dense Max-Margin Loss (DMML) on two classification datasets: **Dermatology** and **Healthcare (Heart Disease)**.

## Project Structure

```
project/
├── README.md
├── requirements.txt
├── training_utilitites.py          # Shared training utilities
├── dermatology/
│   ├── ce_vs_dmml.py               # Dermatology main experiment
│   ├── dermatology.csv             # Dataset
│   └── trial.ipynb                 # Jupyter notebook for exploration
└── healthcare/
    ├── ce_vs_dmml.py               # Healthcare main experiment
    ├── implementation.py            # Shared implementation
    ├── heart.csv                   # Heart disease dataset
    └── healthcare_dataset.csv      # Alternative healthcare dataset
```

## Installation & Setup

### 1. Create and Activate Virtual Environment

```bash
# Navigate to project directory
cd /Users/n_thurai/workspace/comp_6731/project

# Create virtual environment (if not already created)
python3 -m venv project_titanic

# Activate virtual environment
source project_titanic/bin/activate
```

### 2. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

Or individually:
```bash
pip install torch torch-vision torchaudio
pip install scikit-learn pandas numpy matplotlib seaborn
pip install pytest
```

## Running the Experiments

### Run Dermatology Experiment

From the project root directory:

```bash
# Activate virtual environment first
source project_titanic/bin/activate

# Run dermatology CE vs DMML comparison
cd dermatology
python ce_vs_dmml.py
```

**Output:**
- Console logs showing training progress (CE, DMML-Simplified, DMML-Gaussian)
- Training loss plot
- Validation accuracy plot
- PCA visualization of raw features
- t-SNE embeddings for DMML-G and CE models

### Run Healthcare Experiment

From the project root directory:

```bash
# Activate virtual environment first
source project_titanic/bin/activate

# Run healthcare CE vs DMML comparison
cd healthcare
python ce_vs_dmml.py
```

**Output:**
- Console logs showing training progress
- Training loss and accuracy curves
- t-SNE embeddings visualization
- Performance metrics comparison

### Run Both Experiments

```bash
source project_titanic/bin/activate

# Dermatology
cd dermatology && python ce_vs_dmml.py
cd ..

# Healthcare
cd healthcare && python ce_vs_dmml.py
cd ..
```

## Key Features

### Loss Functions Compared

1. **Cross-Entropy (CE)**: Standard classification loss
2. **DMML-Simplified**: Margin-based loss in distance space
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
- **Epochs**: 30
- **Batch Size**: 32 (training), 128 (validation)
- **Train/Val Split**: 80/20 with stratification

### DMML Hyperparameters

- **Margin**: 1.0 (distance margin)
- **Beta**: 0.2 (similarity margin, Gaussian variant)
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

### Paths

The scripts assume they run from their respective dataset directories:

```bash
cd dermatology
python ce_vs_dmml.py  # Looks for dermatology.csv in current directory
```

If you prefer to run from project root, modify the dataset path in `ce_vs_dmml.py`:
```python
# Instead of:
X_train, X_val, y_train, y_val = load_dermatology("dermatology.csv")

# Use:
X_train, X_val, y_train, y_val = load_dermatology("dermatology/dermatology.csv")
```

### Dependencies & Versions

See `requirements.txt` for exact package versions. Key dependencies:
- **PyTorch**: 2.2.0 (deep learning framework)
- **scikit-learn**: 1.7.2 (machine learning utilities)
- **pandas**: 2.2.1 (data manipulation)
- **matplotlib**: 3.8.3 (visualization)
- **numpy**: 1.26.4 (numerical computing)

## Troubleshooting

### Issue: Module not found errors

**Solution**: Ensure you're running from the correct directory or modify import paths:
```python
import sys
sys.path.append('../')  # Add parent directory to path
```

### Issue: "Matplotlib not installed; skipping plots"

**Solution**: Install matplotlib:
```bash
pip install matplotlib
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in the code or disable GPU:
```python
device = torch.device("cpu")  # Force CPU
```

### Issue: CSV file not found

**Solution**: Ensure you're running from the correct directory:
```bash
cd dermatology  # Must be in this directory
python ce_vs_dmml.py
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

1. **Training Loss Curve**: Compares convergence speed of three methods
2. **Validation Accuracy Curve**: Shows final performance
3. **PCA Plot**: 2D projection of raw input features colored by class
4. **t-SNE Embedding**: 2D projection of learned representations, shows cluster separation

Better separated clusters in t-SNE indicate stronger learned representations.

## Virtual Environment Info

The `project_titanic` environment uses Python 3.x with the following key packages:

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
