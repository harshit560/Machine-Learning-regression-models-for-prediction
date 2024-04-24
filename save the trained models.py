import joblib

# Function to save trained models
def save_models(models):
    regr_trans_rf, regr_trans_xg, regr_trans_svr, reg_model_lr, best_model_cubist = models

    # Save Random Forest model
    joblib.dump(regr_trans_rf, 'C:\\Users\\Hp\\Desktop\\random_forest_model.joblib')

    # Save XGBoost model
    joblib.dump(regr_trans_xg, 'C:\\Users\\Hp\\Desktop\\xgboost_model.joblib')

    # Save SVR model
    joblib.dump(regr_trans_svr, 'C:\\Users\\Hp\\Desktop\\svr_model.joblib')

    # Save Linear Regression model
    joblib.dump(reg_model_lr, 'C:\\Users\\Hp\\Desktop\\linear_regression_model.joblib')

    # Save Cubist model
    joblib.dump(best_model_cubist, 'C:\\Users\\Hp\\Desktop\\cubist_model.joblib')

# Save the trained models
save_models(trained_models)