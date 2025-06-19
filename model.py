import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class OptimizedMedicalDataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.categorical_features = ['gender', 'blood_type', 'allergies']
        self.numerical_features = [
            'age', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'temperature'
        ]

    def preprocess(self, data, fit=False):
        processed_data = data.copy()
        for feature in self.categorical_features:
            if feature in processed_data.columns:
                if fit:
                    self.label_encoders[feature] = LabelEncoder()
                    processed_data[feature] = self.label_encoders[feature].fit_transform(processed_data[feature])
                else:
                    processed_data[feature] = self.label_encoders[feature].transform(processed_data[feature])
        return processed_data


class OptimizedTreatmentPredictor:
    def __init__(self):
        self.preprocessor = OptimizedMedicalDataPreprocessor()
        self.model = None
        self.fitted = False

    def train(self, data, target_column):
        processed_data = self.preprocessor.preprocess(data, fit=True)
        X = processed_data.drop(columns=[target_column])
        y = processed_data[target_column]

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, self.preprocessor.numerical_features)
        ])

        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30]
        }

        search = RandomizedSearchCV(
            rf, param_grid, n_iter=5, cv=3,
            n_jobs=-1, random_state=42
        )
        search.fit(X, y)

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', search.best_estimator_)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        self.fitted = True

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig("sample_output.png")
        plt.show()

        return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

    def predict_treatment(self, patient_data: pd.DataFrame):
        if not self.fitted:
            raise Exception("Model must be trained before prediction.")

        processed_data = self.preprocessor.preprocess(patient_data)
        prediction = self.model.predict(processed_data)[0]
        confidence = float(np.max(self.model.predict_proba(processed_data)))

        treatments = {
            0: "Standard Treatment",
            1: "Intensive Care"
        }
        details = {
            0: "Regular monitoring, lifestyle adjustments, and prescribed medication as needed.",
            1: "Close monitoring with advanced support, potential ICU admission, specialized medical intervention."
        }
        guidance = {
            0: "Follow a healthy diet, exercise regularly, and keep regular follow-ups with your doctor.",
            1: "Seek immediate medical attention. A comprehensive evaluation by specialists is recommended."
        }

        return {
            'treatment': treatments[prediction],
            'confidence': confidence,
            'details': details[prediction],
            'guidance': guidance[prediction]
        }
