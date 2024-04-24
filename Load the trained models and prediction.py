import pandas as pd
import json
import joblib

# Load the JSON data
with open('C:\\Users\\Hp\\Desktop\\testing_value.json', 'r') as file:
    input_data = json.load(file)

# Convert JSON data to DataFrame
input_df = pd.DataFrame(input_data)

# Assuming your input data has the same columns as your training data
# Preprocess the input data (scaling)
input_df_norm = (input_df - df.min()) / (df.max() - df.min())
input_df_norm = input_df_norm.drop('prediction_param', axis=1)
# Load trained models
def load_models():
    regr_trans_rf = joblib.load('C:\\Users\\Hp\\Desktop\\random_forest_model.joblib')
    regr_trans_xg = joblib.load('C:\\Users\\Hp\\Desktop\\xgboost_model.joblib')
    regr_trans_svr = joblib.load('C:\\Users\\Hp\\Desktop\\svr_model.joblib')
    reg_model_lr = joblib.load('C:\\Users\\Hp\\Desktop\\linear_regression_model.joblib')
    best_model_cubist = joblib.load('C:\\Users\\Hp\\Desktop\\cubist_model.joblib')
    
    return regr_trans_rf, regr_trans_xg, regr_trans_svr, reg_model_lr, best_model_cubist

# Function to make predictions using the loaded models
def make_predictions(input_df, models):
    regr_trans_rf, regr_trans_xg, regr_trans_svr, reg_model_lr, best_model_cubist = models

    # Making predictions using each model
    yhat_rf = regr_trans_rf.predict(input_df)
    yhat_xg = regr_trans_xg.predict(input_df)
    yhat_svr = regr_trans_svr.predict(input_df)
    yhat_lr = reg_model_lr.predict(input_df)
    yhat_cubist = best_model_cubist.predict(input_df)

    # use different predictions from different models and later destandardise 
    predictions = yhat_cubist[0], yhat_rf[0], yhat_xg[0]
        #'SVR': yhat_svr[0],
        #'LinearRegression': yhat_lr[0],
        
    
    
    return predictions

# Load models
loaded_models = load_models()

# Make predictions
predictions = make_predictions(input_df_norm, loaded_models)

# Print predictions
print(predictions)

from sklearn.preprocessing import QuantileTransformer

# Function to destandardize the prediction
def destandardize_prediction(prediction):
    # Create a QuantileTransformer object
    transformer = QuantileTransformer(output_distribution='normal')
    
    # Fit the transformer on the original target variable (y1)
    transformer.fit(df1)
    
    # Reshape the prediction to match the expected input shape of inverse_transform
    prediction_reshaped = np.array(prediction).reshape(-1, 1)
    
    # De-standardize the prediction
    destandardized_prediction = transformer.inverse_transform(prediction_reshaped)
    
    return destandardized_prediction[0][0]

# De-standardize the median prediction
pred = destandardize_prediction(predictions)

# Print the de-standardized median prediction
print("Prediction:", pred)