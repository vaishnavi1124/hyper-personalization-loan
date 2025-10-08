# 💡 HYPER-PERSONALISATION

## 🧩 Problem Statement
Banks handle large amounts of customer data that must be analyzed to recommend suitable financial products and estimate loan amounts.  
Manual processing is slow, inconsistent, and inefficient.

This project aims to build a **web application** that automatically:
- Segments customers
- Predicts suitable financial products
- Estimates credit limits and loan amounts
- Generates AI-powered customer insights using **Google Gemini**

The goal is to streamline decision-making and enhance customer targeting through **Machine Learning** and **Generative AI**.

---

## 🚀 Features
- 📂 Upload customer CSV data for batch processing  
- 🔍 Predict customer segments using clustering (KMeans)  
- 💳 Recommend financial products:
  - Credit Card  
  - Home Loan  
  - Personal Loan  
- 💰 Estimate loan amounts and credit limits  
- 🧠 Generate concise AI-driven insights using **Gemini API**  
- 📤 Export and view all recommendations  

---

## 🛠️ Technologies Used
- **Python 3**
- **Flask**
- **pandas**
- **scikit-learn**
- **PyMySQL**
- **Google Gemini API**

---

## 📁 Project Structure
```
├── app.py                 
├── models/                
│   ├── kmeans.pkl
│   ├── scaler.pkl
│   ├── multi_target_classifier_model.pkl
│   ├── rf_regressor_credit_card_model.pkl
│   ├── rf_regressor_home_loan_model.pkl
│   └── rf_regressor_personal_loan_model.pkl
├── templates/             
│   ├── home.html
│   ├── index.html
│   └── view_results.html
├── static/                
└── README.md              
```

---

## ⚙️ Setup Instructions

### 1️⃣ Install Dependencies
Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate it:
```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

---

### 2️⃣ Configure Gemini API Key
In your application code, set up your Gemini API key:
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")
```

---

### 3️⃣ Prepare MySQL Database
- Create your database and required tables.  
- Update the database connection details in your Flask app.

---

### 4️⃣ Place Pre-Trained Models
Copy all `.pkl` model files into the `models/` directory:
```
models/
│── kmeans.pkl
│── scaler.pkl
│── multi_target_classifier_model.pkl
│── rf_regressor_credit_card_model.pkl
│── rf_regressor_home_loan_model.pkl
│── rf_regressor_personal_loan_model.pkl
```

---

### 5️⃣ Run the Application
Start the Flask app:
```bash
python app.py
```

Then visit:
```
http://localhost:5000
```

---

## 🔄 How It Works

### 🧮 Data Processing
- Reads uploaded CSV data  
- Preprocesses and scales features  

### 👥 Customer Segmentation
- Predicts segment using **KMeans clustering**

### 🏦 Product Recommendation
- Predicts suitability probabilities for **Credit Card**, **Home Loan**, and **Personal Loan**
- Decision logic:
  - If no probability > 0.5 → No recommendation  
  - If top 2 probabilities are close → Recommend both  
  - Else → Recommend top product

### 💵 Limit Estimation
- Predicts credit limits and loan amounts using **Random Forest Regressors**

### 🧠 Insight Generation
- Generates a concise, 20-word summary using **Google Gemini API**

### 💾 Storage
- Saves all results into the **recommendation_result** table in the database

---

## 🧭 Conclusion
This project demonstrates a complete end-to-end system for:
- Customer segmentation  
- Financial product recommendation  
- AI-generated insights  

By combining **Machine Learning** and **Generative AI**, the system enhances decision-making and delivers personalized recommendations efficiently.  
The intuitive web interface enables easy data uploads, real-time processing, and insightful results — reducing manual work and improving accuracy.

---

## 🖼️ Screenshots
(Add your application screenshots here)

---

## 👩‍💻 Author
**Project Title:** HYPER-PERSONALISATION  
**Domain:** FinTech / AI-Powered Customer Intelligence  
**Developed By:** Vaishnavi Pawar

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
