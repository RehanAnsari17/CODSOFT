***

# Credit Card Fraud Detection

## 📌 Overview
This project implements a **machine learning pipeline** for detecting fraudulent credit card transactions using data from **Kaggle's Fraud Detection dataset**.  
The notebook walks through **data loading, preprocessing, feature engineering, and model training** to predict whether a transaction is fraudulent (`is_fraud = 1`) or legitimate (`is_fraud = 0`).

The task is framed as a **binary classification problem** using structured transaction data.

***

## 📂 Dataset
The dataset used in this project is from:  
**[Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data)**  

It includes:
- `fraudTrain.csv` → Training data  
- `fraudTest.csv` → Testing data  

**Each record contains:**
- **Transaction details** — Amount, time, merchant, category
- **User profile** — Gender, age, location, job, card number
- **Geolocation details** — Latitude & longitude of transaction and merchant
- **Label** — `is_fraud` (1 = Fraud, 0 = Legitimate)

***

## ⚙️ Project Workflow

1. **Import Dependencies**
   - `numpy`, `pandas` for data processing.
   - (Later parts may include `scikit-learn`, `matplotlib`, `xgboost` etc., depending on model choice).

2. **Load Data**
   ```python
   credit_data = pd.read_csv('fraudTrain.csv')
   test_data = pd.read_csv('fraudTest.csv')
   ```

3. **Exploratory Data Analysis (EDA)**
   - Overview of column names, data types, null values.
   - Basic descriptive statistics.
   - Class distribution for `is_fraud` (usually imbalanced).

4. **Feature Engineering**
   - Extract **transaction month** & **year** from `trans_date_trans_time`.
   - Compute age from `dob`.
   - Calculate **distance between transaction location and merchant location**.
   - One-hot encode transaction categories (e.g., `category_entertainment`, `category_food_dining`, etc.).
   - Convert gender from `'M'`/`'F'` to binary `0`/`1`.

5. **Data Preprocessing**
   - Normalize or scale numerical features.
   - Handle categorical variables.
   - Prepare train/test splits.

6. **Model Training**
   - Train machine learning classifiers (e.g., Logistic Regression, Random Forest, Gradient Boosting, XGBoost).
   - Handle **class imbalance** (e.g., SMOTE oversampling or weighted loss).

7. **Model Evaluation**
   - Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**, **ROC-AUC**.
   - Confusion matrix visualization.

8. **Testing on Unseen Data**
   - Final model tested on the test set (`fraudTest.csv`).

***

## 📊 Columns in Dataset (Post-Processing)

| Column Name              | Description |
|--------------------------|-------------|
| `amt`                    | Transaction amount |
| `gender`                 | Gender (0 = Male, 1 = Female) |
| `city_pop`               | Population of customer’s city |
| `age`                    | Customer’s age (in days or years) |
| `trans_month`, `trans_year` | Month & year of transaction |
| `lat_dis`, `long_dis`    | Geographic distance between customer and merchant |
| `category_*`             | One-hot encoded merchant category |
| `is_fraud`               | Target variable (1 = Fraud, 0 = Not Fraud) |

***

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Alternatively, manually install `numpy pandas scikit-learn matplotlib seaborn xgboost`)*

### 3. Download the Dataset
- Go to **Kaggle dataset** page.
- Download `fraudTrain.csv` and `fraudTest.csv`.
- Place them inside the `data/` folder in your project.

### 4. Run Jupyter Notebook
```bash
jupyter notebook CreditFraud.ipynb
```

***

## 📈 Results
The final model achieves:
- **High recall** for fraudulent transactions (important for fraud detection).
- Balanced precision to minimize false positives.
- ROC-AUC around *X.XX* (fill from your results).

Visualizations include:
- Fraud distribution plots.
- ROC curves.
- Feature importance charts.

***

## 🔮 Future Improvements
- Use **deep learning** models like LSTMs for temporal patterns.
- Implement **real-time fraud detection** API with FastAPI/Flask.
- Use **more advanced anomaly detection** methods.


## 🙌 Acknowledgments
- **Kaggle** for providing the dataset.
- **CodSoft** for giving this task.

***


[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/83245422/d1d9fa3a-403a-49cf-bcf9-e2b0ccafb92e/CreditFraud.ipynb
