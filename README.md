# 🏦 Intelligent Automation in Fraud Detection 🚀  

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An integrated Machine Learning and RPA system for real-time financial fraud detection with automatic notification.

## 📌 Objective  
Develop an automated solution that:
1. Analyzes financial transactions using ML models
2. Detects suspicious patterns with high accuracy
3. Executes corrective actions via RPA
4. Sends automatic alerts via email

## 🛠 Technologies Used  
| Category          | Tools                                                                 |
|--------------------|-------------------------------------------------------------------------|
| Language          | Python 3.9+                                                             |
| ML                 | Scikit-learn, XGBoost, Random Forest, SMOTE                           |
| Data Analysis   | Pandas, NumPy                                                           |
| Visualization       | Matplotlib, Seaborn, Plotly                                         |
| Automation          | win32com (Outlook), SMTP                                           |
| Statistics        | Statsmodels (McNemar), SciPy                                         |

## 📂 Project Structure  

```bash
📂 fraud-automation
 ┣ 📂 data
 ┃ ┣ 📂 raw                   # Raw data (not versioned)
 ┃ ┗ 📂 processed             # Pre-processed data
 ┣ 📂 experiments
 ┃ ┣ 📜 train.py              # Training (Credit Card)
 ┃ ┣ 📜 train_ieee.py         # Training (IEEE)
 ┃ ┗ 📜 evaluate.py           # Model evaluation
 ┣ 📂 results
 ┃ ┣ 📂 logs                  # Execution logs
 ┃ ┣ 📂 plots                 # Graphs (ROC, confusion matrix)
 ┃ ┗ 📂 saved_models          # Trained models (.pkl)
 ┣ 📜 main.py                 # Main pipeline
 ┣ 📜 preprocessing.py        # Pre-processing
 ┣ 📜 eda.py                  # Exploratory data analysis
 ┣ 📜 ml_model.py             # Model definitions
 ┣ 📜 email_sending.py        # Email automation
 ┣ 📜 requirements.txt        # Dependencies
 ┗ 📜 README.md               # This file
```

## ⚙️ Setup and Execution

### Prerequisites
- Python 3.9+
- Kaggle account to download datasets

### Installation
```bash
# Clone the repository
git clone https://github.com/your-user/fraud-automation.git
cd fraud-automation

# Install dependencies
pip install -r requirements.txt
```

### Data Download
**IEEE CIS Fraud Dataset:**
- Available on: Kaggle
- Required files: train_transaction.csv, test_transaction.csv
- Save in: `data/raw/ieee/`

**Credit Card Fraud Dataset:**
- Available on: Kaggle
- File: creditcard.csv
- Save in: `data/raw/creditcard/`

### Execution
```bash
# Complete pipeline (pre-processing + training + evaluation)
python main.py

# Train specific models
python experiments/train.py --base creditcard --model xgb
python experiments/train_ieee.py --model rf
```

## 📊 Metrics and Statistics

### Model Comparison
| Model          | Accuracy | Precision | Recall | F1-Score | AUC  |
|---------------|----------|------------|--------|------------|------|
| XGBoost      | 0.998    | 0.952      | 0.821  | 0.881      | 0.983|
| Random Forest | 0.996    | 0.912      | 0.784  | 0.843      | 0.962|

### McNemar Test
```python
# Result:
- Statistic χ²: 4.92
- p-value: 0.027 (p < 0.05 → significant difference)
```

### Generated Graphs
- **Confusion Matrix**
- **ROC Curve**
- **Feature Distribution**

## 📈 Results
- 15% reduction in false negatives vs manual approach
- 40% time savings in analysis via RPA automation
- 95.2% accuracy in fraud detection (XGBoost)

## 📧 Automatic Notification
### Example of generated email:
```plaintext
Subject: [ALERT] Fraud Detected - AntiFraud System

Body:
Three suspicious transactions were identified:
- ID: 12345 | Amount: R$ 2,450.00
- ID: 67890 | Amount: R$ 3,780.50

Attachments:
- fraud_report.xlsx
- analysis_graphs.zip
```

## 📚 References
- Chen, T. (2016). XGBoost: A Scalable Tree Boosting System
- Chawla, N.V. (2002). SMOTE: Synthetic Minority Over-sampling Technique
- Gartner (2021). Market Guide for RPA

## 🔮 Next Steps
- Integration with banking APIs
- Real-time dashboard
- Deep Learning models

## 🤝 Contribution
Contributions are welcome! Follow these steps:

1. Fork the project
2. Create your branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📄 License
Distributed under the MIT license. See LICENSE for more information.

Developed with ❤️ by [Guilherme de Almeida Pereira](https://www.linkedin.com/in/guilhermedealmeidapereira/) | LinkedIn




