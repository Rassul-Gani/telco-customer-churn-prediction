from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

BINARY_COLS = [
    'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
]
CONTRACT_MAP = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
CATEGORICAL_COLS = ['gender', 'InternetService', 'PaymentMethod']


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def preprocess(df):
    pd.set_option('future.no_silent_downcasting', True)

    df = df.drop(columns=['customerID'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])

    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

    for col in BINARY_COLS:
        X[col] = X[col].replace({
            'Yes': 1,
            'No': 0,
            'No internet service': 0,
            'No phone service': 0,
        })

    X['Contract'] = X['Contract'].map(CONTRACT_MAP)
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
    return X, y


def train_and_evaluate(X_train, X_test, y_train, y_test, class_weight=None, label="baseline"):
    model = LogisticRegression(max_iter=5000, class_weight=class_weight)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {label} (class_weight={class_weight}) ===")
    print(f"Recall:   {recall_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=['not churn', 'churn']))

    return model, y_pred


def plot_confusion_matrix(y_test, y_pred, out_path):
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['not churn', 'churn']).plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(model, feature_names, out_path, top_n=15):
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0],
    })
    importance['Abs_Coefficient'] = importance['Coefficient'].abs()
    importance = importance.sort_values('Abs_Coefficient', ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    colors = ['tab:red' if c > 0 else 'tab:blue' for c in importance['Coefficient']]
    plt.barh(importance['Feature'], importance['Coefficient'], color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel('Coefficient (log-odds impact on churn)')
    plt.title(f'Top {top_n} Features Driving Churn Predictions')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    return importance


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_and_evaluate(X_train, X_test, y_train, y_test, class_weight=None, label="Baseline")
    balanced_model, balanced_pred = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_weight='balanced', label="Class-balanced"
    )

    plot_confusion_matrix(y_test, balanced_pred, OUTPUT_DIR / "confusion_matrix.png")
    importance = plot_feature_importance(balanced_model, X_train.columns, OUTPUT_DIR / "feature_importance.png")

    print("\nTop features (class-balanced model):")
    print(importance[['Feature', 'Coefficient']].to_string(index=False))
    print(f"\nPlots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
