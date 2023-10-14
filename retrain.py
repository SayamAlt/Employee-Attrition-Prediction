import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import logging, joblib, warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import BorderlineSMOTE

# Configure the logger
logging.basicConfig(filename='model_retraining.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("model_retraining")
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv",usecols=['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime','YearsAtCompany','TotalWorkingYears','MaritalStatus','StockOptionLevel','Attrition'])
label_mapping = {'No': 0, 'Yes': 1}

def onehotencode(data: pd.DataFrame,col: str) -> pd.DataFrame:
    encoder = OneHotEncoder(drop='first',sparse_output=False,max_categories=10)
    encoded_data = encoder.fit_transform(data[[col]])
    encoded_data = pd.DataFrame(encoded_data,columns=encoder.get_feature_names_out())
    return encoded_data

encode_cols = ['MaritalStatus','OverTime']

for col in encode_cols:
    encoded_data = onehotencode(df,col)
    df = pd.concat([df,encoded_data],axis=1)
    df.drop(col,axis=1,inplace=True)

df.drop('MaritalStatus_Married',axis=1,inplace=True)

df = df[['Age','DailyRate','DistanceFromHome','JobLevel','MonthlyRate','OverTime_Yes','YearsAtCompany','TotalWorkingYears','MaritalStatus_Single','StockOptionLevel','Attrition']]

X = df.drop('Attrition',axis=1)
y = df['Attrition']

smote = BorderlineSMOTE()
X, y = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True,random_state=75)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.map(lambda x: label_mapping[x])
y_test = y_test.map(lambda x: label_mapping[x])

def train_and_evaluate_model(model):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    roc_auc = roc_auc_score(y_test,y_pred)
    logger.info("Evaluation metrics - Accuracy: %.2f, Precision: %.2f, Recall: %.2f, F1 Score: %.2f", accuracy, precision, recall, f1)
    return model, accuracy

model, baseline_acc = train_and_evaluate_model(ExtraTreesClassifier())

param_grid = {'n_estimators': [100,300,600,1000],
             'criterion': ['gini','entropy','log_loss'],
             'max_features': ['auto','sqrt','log2'],
             'bootstrap': [True,False],
             'class_weight': ['balanced','balanced_subsample'],
             'oob_score': [True,False],
             'warm_start': [True,False],
             'max_samples': [0.2,0.4,0.7,1]
             }

grid_et = RandomizedSearchCV(model,param_grid,cv=5,verbose=0)
optimized_model, optimized_acc = train_and_evaluate_model(grid_et)

if baseline_acc < optimized_acc:
    model = optimized_model

avg_cv_scores = cross_val_score(model,X_test,y_test,scoring='accuracy',cv=5,verbose=2)
mean_score = round(np.mean(avg_cv_scores),4) * 100
logger.info("Mean Cross Validation Performance of Extra Trees Classifier: %.2f%",mean_score)

pipeline = Pipeline(steps=[
    ('scaler',scaler),
    ('model',model)
])

logging.shutdown()
joblib.dump(pipeline,'pipeline.pkl')