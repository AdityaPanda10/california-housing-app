import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# Load data
@st.cache_data
def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df

df = load_data()
st.title("🏡 California Housing Price Prediction (ML Project)")
st.write("This app predicts housing prices using Linear & Logistic Regression.")

# Show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Visualization
st.subheader("📊 Data Visualization")

# Histogram
fig1, ax1 = plt.subplots()
sns.histplot(df['MedHouseVal'], bins=30, kde=True, ax=ax1)
st.pyplot(fig1)

# Correlation Heatmap
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# Scatter Plot
fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='MedInc', y='MedHouseVal', ax=ax3)
st.pyplot(fig3)

# ========================
# MODEL TRAINING
# ========================
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

st.subheader("📈 Linear Regression")
st.write("**RMSE:**", round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 3))

# Actual vs Predicted
fig4, ax4 = plt.subplots()
ax4.scatter(y_test[:100], y_pred_lr[:100], alpha=0.7)
ax4.plot([0, 5], [0, 5], '--r')
ax4.set_xlabel("Actual Prices")
ax4.set_ylabel("Predicted Prices")
ax4.set_title("Actual vs Predicted")
st.pyplot(fig4)

# Logistic Regression Setup
def categorize(val):
    if val < 1.5:
        return 0
    elif val < 3:
        return 1
    else:
        return 2

df['PriceCategory'] = df['MedHouseVal'].apply(categorize)
X_cat = df.drop(['MedHouseVal', 'PriceCategory'], axis=1)
y_cat = df['PriceCategory']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_cat, y_train_cat)
y_pred_cat = log_reg.predict(X_test_cat)

st.subheader("📉 Logistic Regression")
st.write("**Accuracy:**", round(accuracy_score(y_test_cat, y_pred_cat)*100, 2), "%")

# Confusion Matrix
fig5, ax5 = plt.subplots()
sns.heatmap(confusion_matrix(y_test_cat, y_pred_cat), annot=True, fmt='d', cmap='Blues', ax=ax5)
ax5.set_xlabel("Predicted")
ax5.set_ylabel("Actual")
ax5.set_title("Confusion Matrix")
st.pyplot(fig5)

# ========================
# USER INPUT + PREDICTION
# ========================
st.header("🎯 Predict Price from Your Inputs")
st.write("Enter the details below:")

medinc = st.number_input("Median Income", min_value=0.0, max_value=20.0, value=3.5)
houseage = st.slider("House Age", 1, 100, 25)
averooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
avebedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
population = st.slider("Population", 100, 5000, 1000)
aveoccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
latitude = st.slider("Latitude", 32.0, 42.0, 34.0)
longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

user_input = pd.DataFrame([[
    medinc, houseage, averooms, avebedrms,
    population, aveoccup, latitude, longitude
]], columns=X.columns)

st.write("Your Input:")
st.dataframe(user_input)

# Linear Prediction
st.subheader("💡 Linear Regression Prediction")
predicted_price = lr.predict(user_input)[0]
st.success(f"🏠 Predicted House Price: **${round(predicted_price, 2)} lakhs**")

fig6, ax6 = plt.subplots()
ax6.scatter(y_test[:100], y_pred_lr[:100], label="Test Data", alpha=0.5)
ax6.scatter(predicted_price, predicted_price, color='red', label="Your Prediction", s=100)
ax6.plot([0, 5], [0, 5], '--r')
ax6.set_xlabel("Actual Prices")
ax6.set_ylabel("Predicted Prices")
ax6.set_title("Your Prediction vs Test")
ax6.legend()
st.pyplot(fig6)

# Logistic Prediction
st.subheader("📌 Logistic Regression (Category Prediction)")
predicted_category = log_reg.predict(user_input)[0]
category_map = {0: "Low", 1: "Medium", 2: "High"}
st.success(f"🏷️ Predicted Price Category: **{category_map[predicted_category]}**")
