import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import optuna

# Load the training and test datasets
df_train = pd.read_csv("fda_trainingset.csv")
df_test = pd.read_csv("fda_testset.csv")

# Define features and target variable
X = df_train.drop(columns=['ID', 'Y'])  # Drop 'ID' column
y = df_train['Y']

# Step 1: Handle missing values using KNN Imputer
knn_imputer = KNNImputer(n_neighbors=10, weights='distance')
X_imputed = knn_imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Step 2: Use LightGBM for Feature Selection
print("\nTraining LightGBM for Feature Selection...")
lgb_fs_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
lgb_fs_model.fit(X, y)

# Get feature importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': lgb_fs_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Select top features (e.g., features with importance > 0)
selected_features = feature_importances[feature_importances['Importance'] > 0]['Feature'].tolist()
print(f"\nSelected Top Features: {selected_features}")

# Reduce the feature set
X_reduced = X[selected_features]

# Initialize AdaBoost model with fixed parameters
adaboost_model = AdaBoostClassifier(
    learning_rate=0.5,
    n_estimators=300,
    random_state=42
)

# Initialize pre-tuned models
xgb_best_params = {
    'max_depth': 14,
    'learning_rate': 0.017777886772191234,
    'n_estimators': 807,
    'gamma': 3.652713784991863,
    'min_child_weight': 7,
    'subsample': 0.6060276329611571,
    'colsample_bytree': 0.8336220511759428,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'random_state': 42
}
xgb_model = xgb.XGBClassifier(**xgb_best_params)

lgbm_best_params = {
    'max_depth': 14,
    'learning_rate': 0.020261475950029136,
    'n_estimators': 320,
    'num_leaves': 89,
    'subsample': 0.8313660657744985,
    'colsample_bytree': 0.62699569878381,
    'min_child_weight': 5,
    'random_state': 42
}
lgbm_model = lgb.LGBMClassifier(**lgbm_best_params)

# Step 3: Create a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('lgbm', lgbm_model),
    ('adaboost', adaboost_model),
    ('xgb', xgb_model)],
    voting='soft', 
    weights=[2, 1, 1.5],
    n_jobs=-1
)

# Train the voting classifier
voting_clf.fit(X_reduced, y)

# Step 4: Prepare the test data
X_test = df_test.drop(columns=['ID'], errors='ignore')
X_test_imputed = knn_imputer.transform(X_test)
X_test_reduced = pd.DataFrame(X_test_imputed, columns=X.columns)[selected_features]

# Predict probabilities on the test set
final_probabilities = voting_clf.predict_proba(X_test_reduced)[:, 1]

# Export predictions to CSV
output = pd.DataFrame({
    'ID': df_test['ID'],
    'Predicted_Probability': final_probabilities
})
output.to_csv('Combination22_Voting_Classifier_LGBM_FS_Preset_XGB_LGBM.csv', index=False)
print("Predictions saved to 'Combination22_Voting_Classifier_LGBM_FS_Preset_XGB_LGBM.csv'.")
