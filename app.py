from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model, scaler, and feature names
try:
    with open('loan_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    feature_names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
                     'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
except Exception as e:
    print(f"Error loading model, feature names, or scaler: {e}")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        Gender = request.form['Gender']
        Married = request.form['Married']
        Education = request.form['Education']
        Self_Employed = request.form['Self_Employed']
        LoanAmount = request.form['LoanAmount']
        Loan_Amount_Term = request.form['Loan_Amount_Term']
        Credit_History = request.form['Credit_History']
        Dependents = request.form['Dependents']
        Property_Area = request.form['Property_Area']
        ApplicantIncome = request.form['ApplicantIncome']
        CoapplicantIncome = request.form['CoapplicantIncome']

        # Create a DataFrame
        data = {
            'Gender': [Gender],
            'Married': [Married],
            'Education': [Education],
            'Self_Employed': [Self_Employed],
            'LoanAmount': [LoanAmount],
            'Loan_Amount_Term': [Loan_Amount_Term],
            'Credit_History': [Credit_History],
            'Dependents': [Dependents],
            'Property_Area': [Property_Area],
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome]
        }
        
        input_data = pd.DataFrame(data)

        # Convert categorical features to numeric
        input_data = input_data.replace({
            'Gender': {'Male': 0, 'Female': 1},
            'Married': {'Yes': 1, 'No': 0},
            'Education': {'Graduate': 1, 'Not Graduate': 0},
            'Self_Employed': {'Yes': 1, 'No': 0},
            'Property_Area': {'Urban': 0, 'Semiurban': 1, 'Rural': 2}
        })

        # Ensure input_data has the same columns as training data
        input_data = input_data.reindex(columns=feature_names, fill_value=0)

        # Standardize input data
        try:
            input_data_scaled = scaler.transform(input_data)
        except Exception as e:
            return f"Error during scaling: {e}"

        # Prediction
        try:
            prediction = model.predict(input_data_scaled)
        except Exception as e:
            return f"Error during prediction: {e}"

        # Determine the result
        if prediction[0] == 1:
            result = 'Approved'
            reasons = []  # No reasons needed if approved
        else:
            result = 'Not Approved'
            reasons = analyze_reasons(input_data.values[0])

        return render_template('result.html', prediction_text=f'Loan Status: {result}', reasons=reasons)

def analyze_reasons(data):
    reasons = []
    
    # Convert data to the correct type
    data = list(map(float, data))  # Convert all elements to float

    # Check feature importance and provide detailed reasons
    if data[5] < 20000:  # ApplicantIncome
        reasons.append("Loan application rejected due to insufficient applicant income. Higher income is required for approval.")

    if data[7] < 360:  # Loan_Amount_Term
        reasons.append("Loan application rejected due to a short loan term. Longer loan terms are preferred for approval.")

    if data[8] == 0:  # Credit_History
        reasons.append("Loan application rejected due to poor credit history. A positive credit history is essential for approval.")

    if data[6] > 500:  # LoanAmount
        reasons.append("Loan application rejected due to the high loan amount requested. Consider requesting a lower amount for approval.")

    if data[2] > 0:  # Dependents
        reasons.append("Loan application rejected due to the number of dependents. Fewer dependents may increase chances of approval.")

    if data[1] == 0:  # Married
        reasons.append("Loan application rejected due to being single. Married applicants are often preferred for loan approvals.")

    if data[10] == 2:  # Property_Area
        reasons.append("Loan application rejected because the property is in a rural area. Urban or semiurban properties are preferred.")

    # Additional checks (less important features)
    if data[9] < 1000:  # CoapplicantIncome
        reasons.append("Loan application rejected due to low coapplicant income. Higher coapplicant income can strengthen the application.")

    if data[4] == 1:  # Self_Employed
        reasons.append("Loan application rejected due to self-employment. Stable employment is typically preferred for loan approvals.")

    return reasons


if __name__ == "__main__":
    app.run(debug=True)
