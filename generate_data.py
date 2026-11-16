"""
Generates a realistic, Telco-style customer churn dataset (7,043 rows).

Why synthetic? It makes the project runnable offline and fully reproducible.
To use the real Kaggle dataset instead, download "WA_Fn-UseC_-Telco-Customer-Churn.csv"
from Kaggle (blastchar/telco-customer-churn), save it as data/telco_churn.csv
and skip this script - train.py works on both.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 7043


def build():
    gender = rng.choice(["Male", "Female"], N)
    senior = rng.binomial(1, 0.16, N)
    partner = rng.choice(["Yes", "No"], N, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", rng.choice(["Yes", "No"], N, p=[0.40, 0.60]),
                          rng.choice(["Yes", "No"], N, p=[0.12, 0.88]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24])
    tenure = np.where(contract == "Month-to-month", rng.gamma(1.6, 12, N),
             np.where(contract == "One year", rng.normal(36, 16, N), rng.normal(52, 18, N)))
    tenure = np.clip(tenure, 1, 72).astype(int)

    phone = rng.choice(["Yes", "No"], N, p=[0.90, 0.10])
    multi = np.where(phone == "No", "No phone service", rng.choice(["Yes", "No"], N))
    internet = rng.choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

    def addon(p_yes=0.45):
        return np.where(internet == "No", "No internet service",
                        rng.choice(["Yes", "No"], N, p=[p_yes, 1 - p_yes]))

    security, backup, protect = addon(0.35), addon(0.42), addon(0.42)
    techsup, tv, movies = addon(0.36), addon(0.50), addon(0.50)

    paperless = rng.choice(["Yes", "No"], N, p=[0.59, 0.41])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                          "Credit card (automatic)"], N, p=[0.34, 0.23, 0.22, 0.21])

    monthly = np.full(N, 18.0)
    monthly = monthly + np.where(phone == "Yes", 7, 0) + np.where(multi == "Yes", 5, 0)
    monthly = monthly + np.where(internet == "DSL", 25, np.where(internet == "Fiber optic", 52, 0))
    for col, price in [(security, 5), (backup, 5), (protect, 5), (techsup, 5), (tv, 9), (movies, 9)]:
        monthly = monthly + np.where(col == "Yes", price, 0)
    monthly = np.round(monthly + rng.normal(0, 2.5, N), 2).clip(18.25, 119)

    total = np.round(monthly * tenure * rng.uniform(0.96, 1.04, N), 2)

    # --- churn probability (logistic model with realistic effects) ---
    z = np.full(N, -2.3)
    z += np.where(contract == "Month-to-month", 1.15, np.where(contract == "One year", -0.20, -1.50))
    z += -0.032 * (tenure - 32)
    z += np.where(internet == "Fiber optic", 0.75, np.where(internet == "No", -0.80, 0.0))
    z += np.where(security == "Yes", -0.45, 0) + np.where(techsup == "Yes", -0.45, 0)
    z += np.where(payment == "Electronic check", 0.55, 0)
    z += np.where(paperless == "Yes", 0.25, 0)
    z += 0.35 * senior - 0.20 * (partner == "Yes") - 0.25 * (dependents == "Yes")
    z += 0.008 * (monthly - 65)
    z += rng.normal(0, 0.55, N)
    churn = rng.binomial(1, 1 / (1 + np.exp(-z)))

    letters = list("ABCDEFGHJKLMNPQRSTUVWXYZ")
    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(letters, 5))}" for _ in range(N)],
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multi, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protect,
        "TechSupport": techsup, "StreamingTV": tv, "StreamingMovies": movies,
        "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    # realistic dirt: blank TotalCharges (as in the real Telco data)
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    idx = rng.choice(df.index, 11, replace=False)
    df.loc[idx, "TotalCharges"] = " "
    return df


if __name__ == "__main__":
    df = build()
    df.to_csv("data/telco_churn.csv", index=False)
    print(df.shape, "| churn rate:", round((df.Churn == "Yes").mean(), 3))
