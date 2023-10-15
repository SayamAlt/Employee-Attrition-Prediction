import pytest, joblib, logging, warnings
warnings.filterwarnings("ignore")
import pandas as pd

pipeline = joblib.load('pipeline.pkl')

# Configure the logger
logging.basicConfig(filename='tests/model_testing.log', level=logging.INFO)

# Create a logger
logger = logging.getLogger("tests/model_testing")
logger.setLevel(logging.DEBUG)  # Set the default log level for the logger

# Create a handler for INFO-level messages
info_handler = logging.FileHandler("tests/model_performance_info.log")
info_handler.setLevel(logging.INFO)

# Create a handler for ERROR-level messages
error_handler = logging.FileHandler("tests/error.log")
error_handler.setLevel(logging.ERROR)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Set the formatter for the handlers
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(info_handler)
logger.addHandler(error_handler)

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
        "input": pd.DataFrame([[37,983,7,4,19603,0.0,10,20.0,1.0,2.0]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 0
    },
    # Test Case 4
    {
        "input": pd.DataFrame([[22,516,21,1,8514,1.0,2,5.0,0.0,1.3]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 1
    },
    # Test Case 5
    {
        "input": pd.DataFrame([[25,809,17,2,11360,1.0,6,13,1.0,0.8]],columns=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel']),
        "expected_output": 1
    }
]

# Create test functions for each test case
@pytest.mark.parametrize("test_input, expected_output", [(tc["input"], tc["expected_output"]) for tc in test_cases])
def test_prediction_with_custom_input(test_input,expected_output):
    # Make a prediction
    prediction = pipeline.predict(test_input)[0]
    logger.debug(prediction)
    try:
        # Check if the prediction matches the expected output
        assert prediction == expected_output
        logger.info("The model's prediction matched the expected output.")
    except:
        logger.error("The model's prediction didn't match with the expected output.")
    
logging.shutdown()