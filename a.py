import pandas as pd
import random
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 Improved Payment On-Time & Date Prediction Dashboard")

# Load CSV directly from file system
csv_file_path = "3000_Varied_Product_Transactions.csv"
df = pd.read_csv(csv_file_path, parse_dates=["date_of_purchase", "payment_due"])

def simulate_payment_date(row): 
    if random.random() < 0.7:
        delta = random.randint(0, 5)
        return row["payment_due"] - timedelta(days=delta)
    else:
        delta = random.randint(1, 30)
        return row["payment_due"] + timedelta(days=delta)

df["payment_date"] = df.apply(simulate_payment_date, axis=1)
df["on_time"] = (df["payment_date"] <= df["payment_due"]).astype(int)
df["days_to_payment"] = (df["payment_due"] - df["date_of_purchase"]).dt.days
df["day_of_week"] = df["date_of_purchase"].dt.dayofweek
df["month"] = df["date_of_purchase"].dt.month
df["is_weekend"] = df["day_of_week"] >= 5
df["is_month_end"] = df["date_of_purchase"].dt.day > 25
df["days_until_payment"] = (df["payment_date"] - df["date_of_purchase"]).dt.days

st.subheader("📁 Sample of Processed Data")
st.dataframe(df.head())

df_model = df.drop(columns=["date_of_purchase", "payment_due", "payment_date", "transact_id"], errors="ignore")
df_model = pd.get_dummies(df_model, drop_first=True)

cols_to_drop = [col for col in df_model.columns if 'customer_id' in col or 'customer_name' in col or 'product_name' in col]
df_model = df_model.drop(columns=cols_to_drop)

# Classification Dataset
df_majority = df_model[df_model.on_time == 1]
df_minority = df_model[df_model.on_time == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X_class = df_balanced.drop("on_time", axis=1)
y_class = df_balanced["on_time"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
clf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 2, 1: 1})
clf_model.fit(X_train_c, y_train_c)
y_pred_c = clf_model.predict(X_test_c)
y_proba_c = clf_model.predict_proba(X_test_c)[:, 1]

threshold = st.slider("Set custom decision threshold", 0.0, 1.0, 0.5, 0.05)
y_pred_adjusted = (y_proba_c >= threshold).astype(int)

st.subheader("📋 Adjusted Classification Report (Custom Threshold)")
report_adj = classification_report(y_test_c, y_pred_adjusted, output_dict=True)
st.dataframe(pd.DataFrame(report_adj).transpose())

st.subheader("📉 Adjusted Confusion Matrix")
cm_adj = confusion_matrix(y_test_c, y_pred_adjusted)
fig1, ax1 = plt.subplots()
sns.heatmap(cm_adj, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Late", "On Time"],
            yticklabels=["Late", "On Time"], ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')
st.pyplot(fig1)

st.subheader("🧪 ROC Curve & AUC")
fpr, tpr, _ = roc_curve(y_test_c, y_proba_c)
auc_score = roc_auc_score(y_test_c, y_proba_c)
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
ax2.plot([0, 1], [0, 1], 'k--')
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend()
st.pyplot(fig2)

st.subheader("📌 Feature Importance (Classification)")
feat_df = pd.DataFrame({
    "feature": X_class.columns,
    "importance": clf_model.feature_importances_
}).sort_values("importance", ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(data=feat_df, x="importance", y="feature", ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)

joblib.dump(clf_model, "improved_payment_model.pkl")
st.success("✅ Improved classification model saved as 'improved_payment_model.pkl'")

# Regression Model
st.subheader("📅 Payment Date Regression")
df_reg = df_model.copy()
df_reg["days_until_payment"] = df["days_until_payment"]

X_reg = df_reg.drop(["on_time", "days_until_payment"], axis=1)
y_reg = df_reg["days_until_payment"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X_train_r, y_train_r)

reg_preds = reg_model.predict(X_test_r)

mae = mean_absolute_error(y_test_r, reg_preds)
mse = mean_squared_error(y_test_r, reg_preds) 
rmse = np.sqrt(mse)
r2 = r2_score(y_test_r, reg_preds)

st.markdown(f"- **MAE**: {mae:.2f} days")
st.markdown(f"- **RMSE**: {rmse:.2f} days")
st.markdown(f"- **R² Score**: {r2:.2f}")

fig4, ax4 = plt.subplots()
sns.scatterplot(x=y_test_r, y=reg_preds, ax=ax4)
ax4.set_xlabel("Actual Days to Payment")
ax4.set_ylabel("Predicted Days to Payment")
ax4.set_title("📅 Actual vs Predicted Payment Timing")
st.pyplot(fig4)

joblib.dump(reg_model, "payment_date_regressor.pkl")
st.success("📦 Regression model saved as 'payment_date_regressor.pkl'")