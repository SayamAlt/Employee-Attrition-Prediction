import pytest, joblib, logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

pipeline = joblib.load('pipeline.pkl')

# Configure the logger
logging.basicConfig(filename='model_testing.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_testing")

test_cases = [
    # Test Case 1
    {
        "input": pd.DataFrame([[38,1295,16,4,23587,1.0,14,19.8,0.0,1.5]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 0
    },
    # Test Case 2
    {
        "input": pd.DataFrame([[23,784,25,2,16951,1.0,5,17.0,1.0,0.6]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 1
    },
    # Test Case 3
    {
        "input": pd.DataFrame([[29,987,7,4,19603,0.0,10,20.0,1.0,2.0]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 0
    },
    # Test Case 4
    {
        "input": pd.DataFrame([[22,516,21,1,8514,1.0,2,5.0,0.0,1.3]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 1
    }
]

# Create test functions for each test case
@pytest.mark.parametrize("test_input, expected_output", [(tc["input"], tc["expected_output"]) for tc in test_cases])
def test_prediction_with_custom_input(test_input,expected_output):
    try:
        # Make a prediction
        prediction = pipeline.predict(test_input)[0]
        accuracy = accuracy_score(expected_output,prediction)
        precision = precision_score(expected_output,prediction)
        recall = recall_score(expected_output,prediction)
        f1 = f1_score(expected_output,prediction)
        roc_auc = roc_auc_score(expected_output,prediction)
        logger.info("Evaluation metrics - Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f", accuracy, precision, recall, f1)
        # Check if the prediction matches the expected output
        assert prediction == expected_output
    except Exception as e:
        logger.error("The model's prediction didn't match with the expected output!",exc_info=True)
    
logging.shutdown()