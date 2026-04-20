# Multi-Feature Model - Quick Start

## Installation & Setup

```bash
# 1. Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install/update dependencies
pip install -r requirements_py313_pytorch26.txt

# 3. Train the model (uses Open, High, Low, Close, Volume)
python train_model_multifeature.py

# 4. Run examples
python example_usage_multifeature.py