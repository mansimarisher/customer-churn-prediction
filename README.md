# Customer Churn Prediction 

A comprehensive machine learning pipeline to predict customer churn, providing actionable insights for business strategy.

##  Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Data Requirements](#-data-requirements)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Business Insights](#-business-insights)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

##  Project Overview

This project implements a complete machine learning pipeline for customer churn prediction. It automatically:

-  **Data Cleaning**: Handles missing values and removes unnecessary columns
-  **Exploratory Data Analysis**: Comprehensive data visualization and statistical analysis
-  **Feature Engineering**: Categorical encoding and numerical scaling
-  **Model Training**: Trains multiple ML models (Random Forest, XGBoost, etc.)
-  **Performance Evaluation**: Detailed metrics and confusion matrices
-  **Business Insights**: Actionable recommendations and ROI analysis
-  **Model Deployment**: Ready-to-use trained models

##  Features

### Data Processing
-  Automatic missing value detection and imputation
-  Smart categorical feature encoding (Label/One-Hot)
-  Feature scaling with StandardScaler
-  Class imbalance handling with SMOTE

### Machine Learning
-  Multiple algorithm comparison (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
-  Automated hyperparameter tuning with GridSearchCV
-  Cross-validation and model selection
-  Feature importance analysis

### Evaluation & Visualization
-  Comprehensive performance metrics (Accuracy, Precision, Recall, F1-Score, AUC)
-  Confusion matrices and ROC curves
-  Feature correlation analysis
-  Business impact visualization

### Business Intelligence
-  Customer risk segmentation (High/Medium/Low)
-  Retention strategy recommendations
-  ROI estimation and business impact analysis

##  Project Structure

```
customer-churn-prediction/
├── data/                           # Data directory
│   └── customer_data.csv          # Your customer dataset (to be added)
├── notebooks/                      # Jupyter notebooks
│   └── customer_churn_prediction.ipynb  # Main analysis notebook
├── models/                         # Trained models
│   ├── churn_prediction_model.pkl  # Final trained model
│   ├── feature_scaler.pkl         # Feature scaler
│   └── feature_names.pkl          # Feature names
├── results/                        # Analysis results
│   ├── model_performance_summary.pkl  # Performance metrics
│   ├── churn_predictions.csv      # Test predictions
│   └── feature_importance.csv     # Feature importance rankings
├── docs/                           # Documentation
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

##  Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd customer-churn-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

4. **Open the main notebook**:
   Navigate to `notebooks/customer_churn_prediction.ipynb`

##  Data Requirements

### Required Format
Your customer data should be a CSV file named `customer_data.csv` placed in the `data/` directory.

### Expected Columns

#### Target Variable (Required)
- **Churn** or **churn** or **CHURN**: Binary column indicating customer churn
  - Accepted formats: `Yes/No`, `1/0`, `True/False`

#### Feature Examples (Typical)
- **customerID**: Unique customer identifier (will be automatically removed)
- **gender**: Customer gender
- **SeniorCitizen**: Senior citizen status
- **Partner**: Has partner (Yes/No)
- **Dependents**: Has dependents (Yes/No)
- **tenure**: Number of months as customer
- **PhoneService**: Has phone service
- **MultipleLines**: Multiple phone lines
- **InternetService**: Internet service type
- **OnlineSecurity**: Online security add-on
- **OnlineBackup**: Online backup add-on
- **DeviceProtection**: Device protection add-on
- **TechSupport**: Technical support add-on
- **StreamingTV**: Streaming TV service
- **StreamingMovies**: Streaming movies service
- **Contract**: Contract type (Month-to-month, One year, Two year)
- **PaperlessBilling**: Uses paperless billing
- **PaymentMethod**: Payment method
- **MonthlyCharges**: Monthly charges
- **TotalCharges**: Total charges to date

### Sample Data Structure

```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,Contract,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,Month-to-month,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,One year,56.95,1889.5,No
3668-QPYBK,Male,0,No,No,2,Yes,Month-to-month,53.85,108.15,Yes
```

##  Usage

### Basic Usage

1. **Prepare your data**: Place `customer_data.csv` in the `data/` directory
2. **Open the notebook**: `notebooks/customer_churn_prediction.ipynb`
3. **Run all cells**: The pipeline will automatically execute end-to-end
4. **Review results**: Check the `results/` directory for outputs

### Advanced Usage

#### Custom Parameters

You can modify key parameters in the notebook:

```python
# Random seed for reproducibility
RANDOM_STATE = 42

# Test set size
test_size = 0.2

# Class imbalance threshold
imbalance_threshold = 0.7

# Model parameters
models = {
    'Random Forest': RandomForestClassifier(n_estimators=200),
    'XGBoost': XGBClassifier(learning_rate=0.1)
}
```

#### Custom Data Processing

```python
# Custom missing value strategy
def custom_imputation(df, column):
    if column in ['tenure', 'MonthlyCharges']:
        return df[column].fillna(df[column].median())
    else:
        return df[column].fillna(df[column].mode()[0])
```

##  Model Performance

### Metrics Tracked
- **Accuracy**: Overall prediction accuracy
- **Precision**: Percentage of predicted churners that actually churned
- **Recall**: Percentage of actual churners correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

### Typical Performance
- **F1-Score**: 0.75-0.85 (varies by dataset)
- **AUC**: 0.80-0.90 (varies by dataset)
- **Accuracy**: 0.80-0.88 (varies by dataset)

### Model Selection
The pipeline automatically selects the best-performing model based on F1-score, considering:
- Logistic Regression (baseline)
- Random Forest (ensemble method)
- Gradient Boosting (sequential ensemble)
- XGBoost (optimized gradient boosting)

##  Business Insights

### Customer Segmentation

The model automatically segments customers into risk categories:

-  **High Risk** (>70% churn probability): Immediate intervention needed
-  **Medium Risk** (30-70% churn probability): Proactive engagement
-  **Low Risk** (<30% churn probability): Standard retention strategies

### Typical Important Features

1. **Contract Type**: Month-to-month contracts show higher churn
2. **Tenure**: Newer customers are more likely to churn
3. **Monthly Charges**: Higher charges correlate with churn risk
4. **Payment Method**: Electronic check payments show higher churn
5. **Total Charges**: Lower lifetime value customers churn more

### ROI Estimation

Example business impact (with $1000 average customer value):
- **Monthly Value Saved**: $50,000 - $100,000
- **Annual Potential Impact**: $600,000 - $1,200,000
- **Intervention Success Rate**: Assumes 50% retention success

##  Deployment

### Saved Model Artifacts

The pipeline saves these files for production use:

- `models/churn_prediction_model.pkl`: Trained model
- `models/feature_scaler.pkl`: Feature scaler
- `models/feature_names.pkl`: Feature names
- `results/model_performance_summary.pkl`: Performance metrics

### Using the Trained Model

```python
import pandas as pd
import joblib

# Load saved models
model = joblib.load('models/churn_prediction_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Make predictions on new data
def predict_churn(new_customer_data):
    # Preprocess the data (same as training)
    processed_data = preprocess_new_data(new_customer_data)
    
    # Scale features
    scaled_data = scaler.transform(processed_data)
    
    # Make prediction
    churn_probability = model.predict_proba(scaled_data)[:, 1]
    churn_prediction = model.predict(scaled_data)
    
    return churn_prediction, churn_probability
```

### Production Considerations

1. **Model Retraining**: Retrain monthly with new data
2. **Performance Monitoring**: Track prediction accuracy vs actual churn
3. **Feature Drift**: Monitor for changes in feature distributions
4. **Threshold Optimization**: A/B test different probability thresholds

##  Model Updates

### Retraining Schedule
- **Frequency**: Monthly
- **Trigger**: Performance drop below 70% F1-score
- **Process**: Re-run entire notebook with new data

### Performance Monitoring
```python
# Monitor model performance
current_performance = calculate_metrics(y_true, y_pred)
if current_performance['f1_score'] < 0.70:
    trigger_retraining()
```

##  Troubleshooting

### Common Issues

1. **File not found error**:
   - Ensure `customer_data.csv` is in the `data/` directory
   - Check file permissions

2. **Memory errors**:
   - Reduce dataset size for testing
   - Use data sampling for large datasets

3. **Package import errors**:
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Model training takes too long**:
   - Reduce hyperparameter grid size
   - Use smaller dataset for testing

### Data Quality Issues

```python
# Check data quality
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Target distribution: {df['Churn'].value_counts()}")
```

##  Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **xgboost**: Gradient boosting framework
- **matplotlib/seaborn**: Data visualization
- **jupyter**: Interactive notebook environment

### Complete List
See `requirements.txt` for full dependency list with version requirements.

##  Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
git clone <repository-url>
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook
```

##  Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the notebook comments and markdown cells
3. Create an issue in the repository

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgments

- Built using scikit-learn and XGBoost
- Inspired by industry best practices in churn prediction
- Designed for business practitioners and data scientists

---


**Ready to predict churn? Place your data in the `data/` directory and run the notebook!** 🚀
