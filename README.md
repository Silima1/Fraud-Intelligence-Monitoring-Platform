# Fraud Intelligence Monitoring Platform

A real-time fraud detection and transaction monitoring system built with Streamlit and machine learning.

## 📁 Project Structure

```
KDD_DASHBOARD/
├── source/                 # Data and model files
│   ├── rf_kaggle.csv      # Model training data
│   ├── rf_kaggle.joblib   # Trained machine learning model
│   ├── test.csv           # Test dataset for predictions
│   └── train.csv          # Training dataset
├── venv/                  # Python virtual environment
├── main.py                # Main Streamlit application
└── README.md              # This file
```

## Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (use requirement.txt)

### Step 1: Environment Setup

1. **Navigate to the project directory:**
   ```bash
   cd KDD_DASHBOARD
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install streamlit pandas numpy scikit-learn matplotlib joblib
```

### Step 3: Verify File Structure

Ensure your project has the following files in the `source/` directory:
- `rf_kaggle.joblib` (trained model)
- `train.csv` (training data)
- `test.csv` (test data)

### Step 4: Run the Application

Launch the Streamlit dashboard:

```bash
streamlit run main.py
```

The application will open in your default web browser at `http://localhost:8501`

##  Features Overview

### 🔍 ID Lookup
- Search and analyze specific transaction IDs
- Real-time fraud probability scoring
- Binary classification: Normal vs Fraud

### 🆕 New Verification
- Manual transaction verification interface
- Custom feature input for testing
- Simulated target testing capabilities

### 📊 Probability Distribution
- Visual analysis of fraud probability across dataset
- Binary classification visualization
- Statistical overview of risk distribution

###  Batch Prediction
- Bulk transaction analysis
- Downloadable risk assessment reports
- Performance metrics and summaries

## 🔧 Technical Details

### Machine Learning Model
- **Algorithm:** Random Forest (from `rf_kaggle.joblib`)
- **Input Features:** Automatically extracted from training data
- **Output:** Binary classification (0 = Normal, 1 = Fraud)
- **Threshold:** 50% probability for unknown targets

### Data Requirements
- Training data must include `fraud_flag` column for supervised learning
- Test data should have compatible feature structure
- Supports temporal features from `application_date` if available

### Customization Options

#### Modify Risk Threshold
Edit the probability threshold in the `get_risk_assessment_binary` function in `main.py`:

```python
# Current threshold for unknown targets
if probability >= 0.5:  # Change this value
    return "🚨 FRAUD", "#DC2626", "High Fraud Probability - Review Required", "fraud"
```

#### Add New Features
Update the feature columns in the data loading section:

```python
def get_feature_columns(clf, train_df):
    # Add or remove feature columns as needed
    target_col = "fraud_flag"
    id_cols = ["ID", "Unnamed: 0", "application_id", "customer_id", "data_batch_id"]
    non_features = [c for c in id_cols + [target_col] if c in train_df.columns]
    return [c for c in train_df.columns if c not in non_features]
```

## 📊 Dashboard Components

### Live Monitoring
- Real-time transaction metrics
- Dynamic trend visualization
- Risk alert system
- Active investigation tracking

### Alert System
- Unusual pattern detection
- International transaction monitoring
- Behavioral anomaly identification
- Volume spike detection

## 🛠️ Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

2. **FileNotFoundError for model or data files**
   - Verify file paths in `main.py`
   - Ensure `source/` directory contains required files

3. **Model compatibility issues**
   - Check scikit-learn version compatibility
   - Retrain model if necessary with current library versions

### Performance Tips

- Use smaller sample sizes for large datasets
- Enable Streamlit caching for repeated operations
- Monitor memory usage with large batch predictions

## 📈 Model Performance

The system uses a pre-trained Random Forest model with the following characteristics:
- Binary classification (Fraud vs Normal)
- Probability-based scoring
- Feature importance ranking
- Real-time prediction capabilities

## 🔒 Security Notes

- This is a demonstration system for fraud detection
- Ensure proper data protection measures in production
- Implement authentication for sensitive financial data
- Follow organizational security protocols for deployment

## 📝 License & Attribution

- **Author:** Leonel Silima
- **Intended Use:** Fraud detection and financial monitoring
- **Model Source:** Kaggle-based training data

##  Contributing

To extend or modify this project:

1. Fork the repository
2. Create a feature branch
3. Test changes with sample data
4. Submit pull request with documentation

## 📞 Support

For technical issues or questions send e-mail to:
1. leonelsilima0@gmail.com


**Note:** This system is designed for educational and demonstration purposes. Always validate with real data and follow organizational guidelines before production deployment.
```

## 📋 Para criar o arquivo README.md:

### Opção 1: Usando o terminal
```bash
# Navegue até a pasta do projeto
cd KDD_DASHBOARD

# Crie o arquivo README.md
cat > README.md << 'EOF'
[cole todo o conteúdo acima aqui]
EOF
```

### Opção 2: Usando um editor de texto
1. Abra seu editor de texto preferido (VSCode, Sublime, Notepad++, etc.)
2. Cole o conteúdo completo acima
3. Salve como `README.md` na pasta principal do projeto `KDD_DASHBOARD/`

### Opção 3: Usando Python
```python
# Execute este código Python no diretório do projeto
readme_content = """
[cole todo o conteúdo do README aqui]
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
```

### Estrutura final do projeto após criar o README:
```
KDD_DASHBOARD/
├── source/
│   ├── rf_kaggle.csv
│   ├── rf_kaggle.joblib
│   ├── test.csv
│   └── train.csv
├── venv/
├── main.py
└── README.md       
  
AUTHOR: Leonel Silima