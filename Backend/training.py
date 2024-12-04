import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from joblib import parallel_backend
from joblib import dump
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

def load_data():
    logging.debug("Loading data...")
    
    df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")
    df = df.drop('id', axis=1)
    
    logging.debug(f"Loaded data shape: {df.shape}")
    
    numerical = ['age', 'avg_glucose_level', 'bmi']
    categorical = ["gender","hypertension", "heart_disease","ever_married","work_type","Residence_type","smoking_status"]
    y = df['stroke']
    X = df.drop('stroke', axis=1)
    return X, y, categorical, numerical

def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    return scores

# Load Data
X, y, categorical, numerical = load_data()

# Check for NaN or problematic data
logging.debug(f"Missing values in X: {X.isnull().sum()}")
logging.debug(f"Unique values in y: {np.unique(y)}")

model = LinearDiscriminantAnalysis()

# Pipeline
transformer = ColumnTransformer(transformers=[
    ("imp", SimpleImputer(strategy="median"), numerical),
    ("o", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

pipeline = Pipeline(steps=[
    ('t', transformer),
    ('p', PowerTransformer(method="yeo-johnson", standardize=True)),
    ('over', SMOTE()),
    ('model', model)
])

# Apply ColumnTransformer
try:
    logging.debug("Applying ColumnTransformer...")
    X_transformed = transformer.fit_transform(X)
    logging.debug(f"Transformed X shape: {X_transformed.shape}")
except Exception as e:
    logging.error(f"Error in ColumnTransformer: {e}")

# Apply PowerTransformer
try:
    logging.debug("Applying PowerTransformer...")
    X_transformed = PowerTransformer(method="yeo-johnson").fit_transform(X_transformed)
    logging.debug("PowerTransformer applied successfully.")
except Exception as e:
    logging.error(f"Error in PowerTransformer: {e}")

# Apply SMOTE
try:
    logging.debug("Applying SMOTE...")
    X_resampled, y_resampled = SMOTE().fit_resample(X_transformed, y)
    logging.debug(f"Resampled X shape: {X_resampled.shape}, Resampled y shape: {y_resampled.shape}")
except Exception as e:
    logging.error(f"Error in SMOTE: {e}")


try:
    logging.debug("Evaluating model...")
    scores = evaluate_model(X, y, pipeline)
    logging.debug(f"Evaluation scores: {scores}")
    print('LDA %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
except Exception as e:
    logging.error(f"Error during model evaluation: {e}")

# Plot the result
# plt.boxplot([scores], label=['LDA'], showmeans=True)
# plt.show()

# Fit the pipeline on entire dataset
try:
    logging.debug("Fitting pipeline...")
    # pipeline.fit(X, y)
    model.fit(X_resampled, y_resampled)
    logging.debug("Pipeline fitted successfully.")
except Exception as e:
    logging.error(f"Error during pipeline fitting: {e}")

# Save the trained pipeline
try:
    logging.debug("Fitting the pipeline on training data...")
    pipeline.fit(X, y)
    logging.debug("Pipeline fitted successfully.")
    
    # Save the trained pipeline
    logging.debug("Saving the pipeline...")
    dump(pipeline, "stroke_pipeline2.joblib")
    logging.debug("Pipeline saved successfully.")
except Exception as e:
    logging.error(f"Error during pipeline fitting or saving: {e}")