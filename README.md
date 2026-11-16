# Customer Churn Prediction - End-to-End Data Science Project

5th Semester Major Project | Python, pandas, scikit-learn, Streamlit

Predicts which telecom customers are likely to leave (churn) so the business can target retention offers.

## Project structure
```
churn_project/
├── data/telco_churn.csv        # dataset (7,043 customers, 20 features)
├── src/generate_data.py        # reproducible dataset generator
├── src/train.py                # EDA -> features -> models -> tuning -> evaluation -> save
├── app.py                      # Streamlit web app for live predictions
├── models/churn_model.joblib   # trained pipeline + decision threshold
├── outputs/metrics.json        # test-set metrics
├── outputs/figures/*.png       # EDA + evaluation charts
├── report/PROJECT_REPORT.md    # full project report
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py     # optional - skip if you use the Kaggle file
python src/train.py             # trains, evaluates, saves model + charts
streamlit run app.py            # opens the web app
```

## Using the real dataset (recommended for submission)
Download **Telco Customer Churn** from Kaggle (IBM sample data, `blastchar/telco-customer-churn`),
save it as `data/telco_churn.csv`, and run `python src/train.py`. No code changes are needed.
The numbers in the report will change slightly - update them from `outputs/metrics.json`.

## Results (held-out test set, 1,409 customers)
| Metric | Value |
|---|---|
| ROC-AUC | 0.844 |
| Recall (churners caught) | 0.818 |
| Precision | 0.496 |
| F1 | 0.617 |
| Accuracy | 0.740 |

Best model: tuned Logistic Regression (beat Random Forest and Gradient Boosting in 5-fold CV).
Top drivers: Contract type, tenure, average monthly spend, internet service, payment method.
