# 📊 Customer Churn Predictor

A Machine Learning web application that predicts whether a telecom customer is likely to **churn (leave)** or **stay**, built with Logistic Regression and deployed using Streamlit.

🔗 **Live Demo:** [Click here to try the app](https://your-app-link.streamlit.app)

---

## 🚀 Features

- Predicts customer churn probability in real-time
- Interactive form with 18+ customer attributes
- Visual churn risk level (Low / Medium / High)
- Clean 3-column UI layout with instant results
- Deployed publicly on Streamlit Cloud

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| Scikit-learn | Logistic Regression model |
| Streamlit | Web app framework |
| Pandas & NumPy | Data processing |
| Joblib | Model serialization |
| Jupyter Notebook | Model training & EDA |

---

## 📁 Project Structure

```
Customer-Churn-Predictor/
├── app.py                    # Streamlit web application
├── best_model.pkl            # Trained Logistic Regression model
├── requirements.txt          # Python dependencies
├── Customer_churn_pr.ipynb   # Model training notebook (EDA + ML)
└── Telco-Customer-Churn.csv  # Dataset
```

---

## 📊 Dataset

**Telco Customer Churn Dataset**
- 7,043 customer records
- 21 features including demographics, services, and billing info
- Target variable: `Churn` (Yes / No)

---

## 🧠 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Logistic Regression |
| Features | 44 (after one-hot encoding) |
| Classes | Churn (1) / No Churn (0) |
| Max Iterations | 1000 |
| Solver | lbfgs |

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/fardeenkhan-tech/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📸 App Preview

> Fill in customer details → Click Predict → Get instant churn risk result with probability score

---

## 👨‍💻 Author

**Fardeen Khan**
- 🎓 B.Tech CSE (2022–2026)
- 💼 Data Science & GenAI Trainee — Ducat India
- 🔗 [GitHub](https://github.com/fardeenkhan-tech)

---

## 📄 License

This project is licensed under the MIT License.
