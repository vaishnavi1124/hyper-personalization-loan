from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import re
import io
import pymysql
from datetime import date

# Gemini import and configuration
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

genai.configure(api_key="AIzaSyCtATLQV8Mcm4NdW50EOxMoIgflvzBoBYU")

app = Flask(__name__)

model_paths = {
    'kmeans': r'C:\Users\vaishnavi pawar\Desktop\models\kmeans.pkl',
    'scaler': r'C:\Users\vaishnavi pawar\Desktop\models\scaler.pkl',
    'multi_target_classifier': r'C:\Users\vaishnavi pawar\Desktop\models\multi_target_classifier_model.pkl',
    'rf_regressor_personal_loan': r'C:\Users\vaishnavi pawar\Desktop\models\rf_regressor_personal_loan_model.pkl',
    'rf_regressor_home_loan': r'C:\Users\vaishnavi pawar\Desktop\models\rf_regressor_home_loan_model.pkl',
    'rf_regressor_credit_card': r'C:\Users\vaishnavi pawar\Desktop\models\rf_regressor_credit_card_model.pkl'
}

# Load models
kmeans = joblib.load(model_paths['kmeans'])
scaler = joblib.load(model_paths['scaler'])
multi_target_classifier = joblib.load(model_paths['multi_target_classifier'])
rf_regressor_personal_loan = joblib.load(model_paths['rf_regressor_personal_loan'])
rf_regressor_home_loan = joblib.load(model_paths['rf_regressor_home_loan'])
rf_regressor_credit_card = joblib.load(model_paths['rf_regressor_credit_card'])

def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        database='recommendation',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

def generate_insights(customer_data):
    prompt = f"""Generate a personalized summarised insight about the following customer in easy to understand language based on their data the response should not more than 20 words:

- Name: {customer_data.get('name', 'Unknown')}
- Age: {customer_data.get('age')}
- Gender: {customer_data.get('gender')}
- Marital Status: {customer_data.get('marital_status')}
- Education: {customer_data.get('education')}
- Occupation: {customer_data.get('occupation')}
- Salary: ${customer_data.get('salary', 0):,.2f}
- Loan Amount: ${customer_data.get('loan_amount', 0):,.2f}
- Credit Limit: ${customer_data.get('credit_limit', 0):,.2f}
- Credit Utilization: {customer_data.get('credit_utilization', 0):.2%}
- EMI Paid: {customer_data.get('emi_paid')}
- Tenure Months: {round(float(customer_data.get('tenure_months', 0)),2)}
- Max DPD: {customer_data.get('max_dpd')}
- Default Status: {int(customer_data.get('default_status',0))}
- Account Balance: ${customer_data.get('account_balance',0):,.2f}
- Credit Card: {customer_data.get('Credit Card')}
- Home Loan: {customer_data.get('Home Loan')}
- Personal Loan: {customer_data.get('Personal Loan')}

Here are the Summarised Insights about {customer_data.get('name','this customer')}: """

    model = genai.GenerativeModel(
        model_name="models/gemini-1.5-pro",
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    response = model.generate_content(prompt)
    return response.text

def clean_and_extract_insight(insights):
    cleaned_insight = re.sub(r'[^a-zA-Z0-9\s]', '', insights)
    if "Here are the Summarised Insights about" in cleaned_insight:
        extracted_insight = cleaned_insight.split("Here are the Summarised Insights about")[1].strip().split("\n\n")[0]
    else:
        extracted_insight = cleaned_insight.strip()
    return extracted_insight

def process_customer_data(json_data, scaler):
    customer_data = pd.DataFrame.from_dict(json_data, orient='index').T
    expected_cols = [
        'age', 'salary', 'loan_amount', 'credit_limit', 'credit_utilization',
        'emi_paid', 'tenure_months', 'max_dpd', 'default_status',
        'enquiry_amount', 'unique_products_enquired', 'total_enquiries',
        'transaction_amount', 'account_balance', 'is_salary',
        'Credit Card', 'Home Loan', 'Personal Loan'
    ]
    for col in expected_cols:
        if col not in customer_data.columns:
            customer_data[col] = 0
    clustering_data = customer_data[expected_cols].fillna(0)
    scaled_data = scaler.transform(clustering_data)
    return customer_data, scaled_data

def predict_customer_segment(scaled_data, kmeans):
    return kmeans.predict(scaled_data)[0]

def recommend_product_and_loan(json_data, kmeans, scaler, multi_target_classifier,
                               rf_regressor_personal_loan, rf_regressor_home_loan, rf_regressor_credit_card):
    customer_data, scaled_data = process_customer_data(json_data, scaler)
    customer_segment = predict_customer_segment(scaled_data, kmeans)
    customer_data['customer_segment'] = customer_segment
    customer_data = customer_data[['age', 'salary', 'loan_amount', 'credit_limit', 'credit_utilization',
                                   'emi_paid', 'tenure_months', 'max_dpd', 'default_status',
                                   'enquiry_amount', 'unique_products_enquired', 'total_enquiries',
                                   'transaction_amount', 'account_balance', 'is_salary',
                                   'Credit Card', 'Home Loan', 'Personal Loan', 'customer_segment']]
    X_classification_prod = customer_data.drop(columns=['Credit Card', 'Home Loan', 'Personal Loan'])
    X_classification_amt = customer_data

    prob_credit_card = [est.predict_proba(X_classification_prod)[:, 1] for est in multi_target_classifier.estimators_]
    product_probabilities = pd.Series({
        'Credit Card': prob_credit_card[0][0],
        'Home Loan': prob_credit_card[1][0],
        'Personal Loan': prob_credit_card[2][0]
    })

    sorted_probs = product_probabilities.sort_values(ascending=False)
    top_product = sorted_probs.index[0]
    top_prob = sorted_probs.iloc[0]
    second_product = sorted_probs.index[1]
    second_prob = sorted_probs.iloc[1]

    predictions = {
        'Credit Card': rf_regressor_credit_card.predict(X_classification_amt.drop(columns=['loan_amount']))[0],
        'Home Loan': rf_regressor_home_loan.predict(X_classification_amt.drop(columns=['loan_amount']))[0],
        'Personal Loan': rf_regressor_personal_loan.predict(X_classification_amt.drop(columns=['loan_amount']))[0]
    }

    recommendation_text = ""
    recommended_product_names = ""

    if top_prob < 0.5:
        recommendation_text = "No suitable product recommendations found for this customer."
        recommended_product_names = "No Recommendation"
    else:
        if (top_prob - second_prob) < 0.09:
            recommendation_text = (
                f"Recommended Products:\n"
                f"- {top_product} (Probability: {top_prob:.2f})\n"
                f"- {second_product} (Probability: {second_prob:.2f})"
            )
            recommended_product_names = f"{top_product}, {second_product}"
        else:
            recommendation_text = (
                f"Recommended Product: {top_product} (Probability: {top_prob:.2f})"
            )
            recommended_product_names = top_product

    recommendation_text += (
        f"\n\nProduct Limits:\n"
        f"- Predicted Credit Card Limit: {predictions['Credit Card']:,.2f}\n"
        f"- Predicted Home Loan Amount: {predictions['Home Loan']:,.2f}\n"
        f"- Predicted Personal Loan Amount: {predictions['Personal Loan']:,.2f}"
    )

    return recommendation_text, recommended_product_names, customer_segment, product_probabilities, predictions

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/export_customers')
def export_customers():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM customer_master")
        rows = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(rows)
    csv_str = df.to_csv(index=False)
    csv_bytes = csv_str.encode('utf-8')
    csv_file = io.BytesIO(csv_bytes)
    return send_file(
        csv_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name='customer_master.csv'
    )

@app.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    all_results = []
    if request.method == 'POST':
        if 'process_csv' in request.form and 'csv_file' in request.files:
            file = request.files['csv_file']
            if file and file.filename.endswith('.csv'):
                df = pd.read_csv(file)
                for idx, row in df.iterrows():
                    json_data = row.to_dict()
                    cust_id = int(json_data.get("id", 0))

                    conn = get_db_connection()
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT cust_name FROM customer_master WHERE id = %s", (cust_id,))
                        result = cursor.fetchone()
                    cust_name = result["cust_name"] if result and result["cust_name"] else "Unknown"
                    conn.close()

                    recommendation, recommended_product_names, customer_segment, product_probabilities, predictions = recommend_product_and_loan(
                        json_data, kmeans, scaler, multi_target_classifier,
                        rf_regressor_personal_loan, rf_regressor_home_loan, rf_regressor_credit_card
                    )
                    insights_raw = generate_insights(json_data)
                    cleaned_insight = clean_and_extract_insight(insights_raw)
                    insights = [line.strip() for line in cleaned_insight.split("\n") if line.strip()]

                    conn = get_db_connection()
                    with conn.cursor() as cursor:
                        sql_insert_result = """
                            INSERT INTO recommendation_result (
                                cust_id, run_date, cust_name, cust_segment,
                                credit_card_probability, home_loan_probability, personal_loan_probability,
                                credit_card_limit, home_loan_limit, personal_loan_limit,
                                recommended_product, insight
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        cursor.execute(
                            sql_insert_result,
                            (
                                cust_id,
                                date.today(),
                                cust_name,
                                int(customer_segment),
                                float(product_probabilities["Credit Card"]) * 100,
                                float(product_probabilities["Home Loan"]) * 100,
                                float(product_probabilities["Personal Loan"]) * 100,
                                float(predictions["Credit Card"]),
                                float(predictions["Home Loan"]),
                                float(predictions["Personal Loan"]),
                                recommended_product_names,
                                " ".join(insights)
                            )
                        )
                    conn.commit()
                    conn.close()

                    all_results.append({
                        "name": cust_name,
                        "recommendation": recommendation,
                        "customer_segment": customer_segment,
                        "product_probabilities": product_probabilities,
                        "insights": insights
                    })
    return render_template('index.html', all_results=all_results)

@app.route('/view_results', methods=['GET', 'POST'])
def view_results():
    conn = get_db_connection()
    run_dates = []
    segment_counts = []
    selected_date = None
    all_results = []

    with conn.cursor() as cursor:
        cursor.execute("SELECT DISTINCT run_date FROM recommendation_result ORDER BY run_date DESC")
        run_dates = [row["run_date"] for row in cursor.fetchall()]

    with conn.cursor() as cursor:
        cursor.execute("SELECT * FROM recommendation_result ORDER BY run_date DESC, cust_id")
        all_results = cursor.fetchall()

    if request.method == 'POST':
        selected_date = request.form.get('run_date')
        if selected_date:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT cust_segment, COUNT(*) AS customer_count
                    FROM recommendation_result
                    WHERE run_date = %s
                    GROUP BY cust_segment
                    ORDER BY cust_segment
                    """,
                    (selected_date,)
                )
                segment_counts = cursor.fetchall()
    conn.close()
    return render_template(
        'view_results.html',
        run_dates=run_dates,
        selected_date=selected_date,
        segment_counts=segment_counts,
        all_results=all_results
    )

if __name__ == '__main__':
    app.run(debug=True)
