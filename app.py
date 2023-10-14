from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

pipeline = joblib.load('pipeline.pkl')
label_mapping = {0: 'No', 1: 'Yes'}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = request.form['age']
        daily_rate = request.form['daily_rate']
        distance_from_home = request.form['distance_from_home']
        job_level = request.form['job_level']
        monthly_rate = request.form['monthly_rate']
        bool_mapping = {'Yes': 1, 'No': 0}
        overtime_yes = bool_mapping[request.form['overtime_yes']]
        years_at_company = request.form['years_at_company']
        total_working_years = request.form['total_working_years']
        marital_status_single = bool_mapping[request.form['marital_status_single']]
        stock_option_level = request.form['stock_option_level']

        data = [[age,daily_rate,distance_from_home,job_level,monthly_rate,overtime_yes,years_at_company,total_working_years,marital_status_single,stock_option_level]]
        pred = pipeline.predict(data)[0]
        pred = label_mapping[pred]

        if pred == 'No':
            return render_template('index.html',prediction_text="The employee with the specified details will stay in the company.")
        elif pred == 'Yes':
            return render_template('index.html',prediction_text="The employee with the specified details will leave the company.")

if __name__ == '__main__':
    app.run(port=5000)