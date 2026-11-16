"""
End-to-end pipeline: load -> clean -> EDA -> feature engineering -> model comparison
-> tuning -> evaluation -> save model + metrics + figures.

Run from the project root:  python src/train.py
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


# ------------------------------------------------------------------ 1. LOAD & CLEAN
def load_and_clean():
    df = pd.read_csv(ROOT / "data" / "telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.drop(columns=["customerID"]).drop_duplicates()
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


# ------------------------------------------------------------------ 2. EDA
def run_eda(df):
    # target balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Target distribution"); ax.set_ylabel("Customers"); plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(FIG / "01_target_distribution.png", dpi=150); plt.close(fig)

    # churn rate by key categoricals
    cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, c in zip(axes.ravel(), cats):
        rate = df.groupby(c)["Churn"].mean().sort_values(ascending=False) * 100
        sns.barplot(x=rate.values, y=rate.index, ax=ax, color="#e76f51")
        ax.set_title(f"Churn rate by {c}"); ax.set_xlabel("Churn %"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(FIG / "02_churn_by_category.png", dpi=150); plt.close(fig)

    # numeric distributions
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, c in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(data=df, x=c, hue=df["Churn"].map({0: "Stayed", 1: "Churned"}), fill=True, ax=ax,
                    palette=["#2a9d8f", "#e76f51"], common_norm=False)
        ax.set_title(c)
    fig.tight_layout(); fig.savefig(FIG / "03_numeric_distributions.png", dpi=150); plt.close(fig)

    # correlation heatmap
    num = df.select_dtypes("number")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation (numeric features)")
    fig.tight_layout(); fig.savefig(FIG / "04_correlation.png", dpi=150); plt.close(fig)


# ------------------------------------------------------------------ 3. FEATURES
def add_features(df):
    df = df.copy()
    df["tenure_group"] = pd.cut(df["tenure"], [0, 12, 24, 48, 72], labels=["0-1y", "1-2y", "2-4y", "4-6y"]).astype(str)
    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_addons"] = (df[addons] == "Yes").sum(axis=1)
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def make_preprocessor(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat),
    ])


# ------------------------------------------------------------------ 4. MODELLING
def main():
    df = load_and_clean()
    print(f"Loaded {df.shape[0]} rows | churn rate {df.Churn.mean():.1%}")
    run_eda(df)
    df = add_features(df)

    X, y = df.drop(columns="Churn"), df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                min_samples_leaf=5, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }

    print("\n5-fold CV (ROC-AUC) on training set")
    rows = []
    for name, est in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor(X)), ("model", est)])
        s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_auc_mean": s.mean(), "cv_auc_std": s.std()})
        print(f"  {name:20s} {s.mean():.4f} +/- {s.std():.4f}")
    cv_table = pd.DataFrame(rows)

    # hyper-parameter tuning on the strongest candidate
    best_name = cv_table.sort_values("cv_auc_mean", ascending=False).iloc[0]["model"]
    print(f"\nTuning: {best_name}")
    pipe = Pipeline([("prep", make_preprocessor(X)), ("model", candidates[best_name])])
    grids = {
        "Logistic Regression": {"model__C": np.logspace(-2, 1.5, 12)},
        "Random Forest": {"model__max_depth": [6, 8, 12, None], "model__min_samples_leaf": [3, 5, 10, 20],
                          "model__max_features": ["sqrt", 0.3, 0.5]},
        "Gradient Boosting": {"model__n_estimators": [100, 200, 300], "model__learning_rate": [0.03, 0.05, 0.1],
                              "model__max_depth": [2, 3, 4], "model__subsample": [0.7, 0.85, 1.0]},
    }
    search = RandomizedSearchCV(pipe, grids[best_name], n_iter=12, scoring="roc_auc", cv=cv,
                                random_state=SEED, n_jobs=-1)
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    print("  best params:", search.best_params_, f"| CV AUC {search.best_score_:.4f}")

    # decision threshold: maximise F1 on out-of-fold training predictions (no test leakage)
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    threshold = float(t[np.argmax(f1s)])
    print(f"  chosen threshold: {threshold:.2f}")

    # ---------------------------------------------------------- 5. EVALUATION (held-out test)
    proba = best.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "best_model": best_name,
        "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "threshold": round(threshold, 3),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
        "cv_table": cv_table.round(4).to_dict("records"),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "churn_rate": round(float(y.mean()), 4),
    }
    print("\nTEST SET:", {k: metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]})

    # baseline comparison for the report
    base = Pipeline([("prep", make_preprocessor(X)), ("model", LogisticRegression(max_iter=2000))]).fit(X_tr, y_tr)
    metrics["baseline_logreg_auc"] = round(roc_auc_score(y_te, base.predict_proba(X_te)[:, 1]), 4)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Stayed", "Churned"], cmap="Blues", ax=ax)
    ax.set_title("Confusion matrix (test)"); ax.grid(False)
    fig.tight_layout(); fig.savefig(FIG / "05_confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    RocCurveDisplay.from_predictions(y_te, proba, name=best_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("ROC curve (test)")
    fig.tight_layout(); fig.savefig(FIG / "06_roc_curve.png", dpi=150); plt.close(fig)

    # permutation importance on ORIGINAL columns (so it is interpretable)
    imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=8, random_state=SEED, n_jobs=-1)
    imp_df = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=imp_df.values, y=imp_df.index, color="#264653", ax=ax)
    ax.set_title("Top 10 churn drivers (permutation importance)"); ax.set_xlabel("Drop in ROC-AUC when shuffled")
    fig.tight_layout(); fig.savefig(FIG / "07_feature_importance.png", dpi=150); plt.close(fig)
    metrics["top_features"] = {k: round(float(v), 4) for k, v in imp_df.items()}

    # ---------------------------------------------------------- 6. SAVE
    joblib.dump({"model": best, "threshold": threshold, "columns": X.columns.tolist()}, ROOT / "models" / "churn_model.joblib")
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved model, metrics.json and figures.")


if __name__ == "__main__":
    main()
