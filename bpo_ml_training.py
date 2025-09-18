#!/usr/bin/env python3

"""
BPO Claims ML Training Pipeline

Train ML models using the labeled BPO claims dataset to predict:
1. Claim approval/rejection (Binary Classification)
2. Specific rejection reasons (Multi-class Classification)
3. Approval probability scores (Regression)

This enables the BPO to screen claims BEFORE submission to maximize pay rates.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BPOMLTrainer:
    def __init__(self, data_file: str = 'bpo_claims_training_dataset.csv'):
        """Initialize ML training pipeline"""
        self.data_file = data_file
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}

    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load training data and prepare for ML"""
        logger.info(f"Loading training data from {self.data_file}")

        try:
            df = pd.read_csv(self.data_file)
            logger.info(f"Loaded {len(df):,} training records")
        except FileNotFoundError:
            logger.error(f"Training data file not found: {self.data_file}")
            logger.info("Run 'python bpo_training_dataset.py' to generate training data first")
            raise

        # Convert datetime columns
        date_columns = ['service_date', 'submission_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Handle missing values
        df = df.fillna(0)

        return df

    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare feature matrix for ML training"""

        # Select numerical features
        numerical_features = [
            'patient_age', 'days_service_to_submission', 'submission_day_of_week',
            'submission_hour', 'billed_amount', 'expected_reimbursement', 'amount_per_age',
            'coder_experience_years', 'coding_accuracy_score', 'documentation_enhancement_score',
            'processing_time_hours', 'provider_data_completeness', 'missing_documentation_count',
            'provider_response_time_hours', 'provider_historical_approval_rate',
            'patient_previous_rejections', 'bpo_coder_approval_rate'
        ]

        # Select categorical features
        categorical_features = [
            'patient_gender', 'patient_state', 'primary_diagnosis_code', 'procedure_code',
            'claim_type', 'provider_specialty', 'primary_payor', 'network_status', 'payor_tier'
        ]

        # Select boolean features
        boolean_features = [
            'prior_authorization_obtained', 'eligibility_verification_completed',
            'medical_necessity_documented', 'qa_review_completed', 'high_dollar_claim',
            'complex_diagnosis', 'weekend_submission', 'after_hours_submission'
        ]

        # Prepare feature matrix
        feature_df = pd.DataFrame()

        # Add numerical features
        for feat in numerical_features:
            if feat in df.columns:
                feature_df[feat] = df[feat]

        # Add boolean features (convert to int)
        for feat in boolean_features:
            if feat in df.columns:
                feature_df[feat] = df[feat].astype(int)

        # Encode categorical features
        for feat in categorical_features:
            if feat in df.columns:
                # Use frequency encoding for high-cardinality categories
                if df[feat].nunique() > 10:
                    freq_map = df[feat].value_counts().to_dict()
                    feature_df[f"{feat}_frequency"] = df[feat].map(freq_map)
                else:
                    # One-hot encoding for low-cardinality categories
                    dummies = pd.get_dummies(df[feat], prefix=feat)
                    feature_df = pd.concat([feature_df, dummies], axis=1)

        # Add derived features if they exist
        derived_features = ['bpo_quality_score', 'provider_risk_score', 'claim_complexity_score']
        for feat in derived_features:
            if feat in df.columns:
                feature_df[feat] = df[feat]

        logger.info(f"Prepared {len(feature_df.columns)} features for training")

        return feature_df, feature_df.columns.tolist()

    def train_approval_classifier(self, X: pd.DataFrame, y_approval: pd.Series) -> dict:
        """Train binary classifier for approval prediction"""
        logger.info("Training approval prediction model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_approval, test_size=0.2, random_state=42, stratify=y_approval
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest (best for tabular data)
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)

        # Train Logistic Regression (interpretable baseline)
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        lr_model.fit(X_train_scaled, y_train)

        # Evaluate models
        rf_predictions = rf_model.predict(X_test_scaled)
        lr_predictions = lr_model.predict(X_test_scaled)

        rf_accuracy = accuracy_score(y_test, rf_predictions)
        lr_accuracy = accuracy_score(y_test, lr_predictions)

        logger.info(f"Random Forest accuracy: {rf_accuracy:.3f}")
        logger.info(f"Logistic Regression accuracy: {lr_accuracy:.3f}")

        # Select best model
        if rf_accuracy >= lr_accuracy:
            best_model = rf_model
            best_accuracy = rf_accuracy
            best_predictions = rf_predictions
            model_type = "Random Forest"
        else:
            best_model = lr_model
            best_accuracy = lr_accuracy
            best_predictions = lr_predictions
            model_type = "Logistic Regression"

        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True))
        else:
            feature_importance = dict(zip(X.columns, abs(best_model.coef_[0])))
            feature_importance = dict(sorted(feature_importance.items(),
                                           key=lambda x: x[1], reverse=True))

        # Store models and scalers
        self.models['approval_classifier'] = best_model
        self.scalers['approval_classifier'] = scaler
        self.feature_importance['approval_classifier'] = feature_importance

        return {
            'model_type': model_type,
            'accuracy': best_accuracy,
            'predictions': best_predictions,
            'actual': y_test,
            'classification_report': classification_report(y_test, best_predictions),
            'feature_importance': feature_importance
        }

    def train_rejection_reason_classifier(self, X: pd.DataFrame, df: pd.DataFrame) -> dict:
        """Train multi-class classifier for rejection reasons"""
        logger.info("Training rejection reason prediction model...")

        # Only use rejected claims for this model
        rejected_mask = ~df['is_approved']
        X_rejected = X[rejected_mask]
        y_rejection = df[rejected_mask]['rejection_reason']

        if len(X_rejected) == 0:
            logger.warning("No rejected claims found for training rejection reason model")
            return {}

        # Encode rejection reasons
        le = LabelEncoder()
        y_rejection_encoded = le.fit_transform(y_rejection)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_rejected, y_rejection_encoded, test_size=0.2, random_state=42, stratify=y_rejection_encoded
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train_scaled, y_train)

        # Evaluate
        predictions = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, predictions)

        logger.info(f"Rejection reason classifier accuracy: {accuracy:.3f}")

        # Feature importance
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True))

        # Store models
        self.models['rejection_reason_classifier'] = rf_model
        self.scalers['rejection_reason_classifier'] = scaler
        self.encoders['rejection_reason_encoder'] = le
        self.feature_importance['rejection_reason_classifier'] = feature_importance

        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'actual': y_test,
            'class_names': le.classes_,
            'feature_importance': feature_importance
        }

    def train_probability_regressor(self, X: pd.DataFrame, y_probability: pd.Series) -> dict:
        """Train regression model for approval probability prediction"""
        logger.info("Training approval probability prediction model...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_probability, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest Regressor
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)

        # Evaluate
        predictions = rf_model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, predictions)

        logger.info(f"Probability regressor MAE: {mae:.4f}")

        # Feature importance
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        feature_importance = dict(sorted(feature_importance.items(),
                                       key=lambda x: x[1], reverse=True))

        # Store models
        self.models['probability_regressor'] = rf_model
        self.scalers['probability_regressor'] = scaler
        self.feature_importance['probability_regressor'] = feature_importance

        return {
            'mae': mae,
            'predictions': predictions,
            'actual': y_test,
            'feature_importance': feature_importance
        }

    def train_all_models(self):
        """Train all ML models"""
        logger.info("Starting comprehensive ML training pipeline...")

        # Load and prepare data
        df = self.load_and_prepare_data()
        X, feature_names = self.prepare_features(df)

        # Train approval classifier
        approval_results = self.train_approval_classifier(X, df['is_approved'])

        # Train rejection reason classifier
        rejection_results = self.train_rejection_reason_classifier(X, df)

        # Train probability regressor
        probability_results = self.train_probability_regressor(X, df['approval_probability_actual'])

        # Save models
        self.save_models()

        # Generate training report
        self.generate_training_report(approval_results, rejection_results, probability_results)

        return {
            'approval_results': approval_results,
            'rejection_results': rejection_results,
            'probability_results': probability_results
        }

    def save_models(self):
        """Save trained models and preprocessors"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model_name, model in self.models.items():
            filename = f"bpo_{model_name}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"Saved {model_name} to {filename}")

        for scaler_name, scaler in self.scalers.items():
            filename = f"bpo_{scaler_name}_scaler_{timestamp}.joblib"
            joblib.dump(scaler, filename)

        for encoder_name, encoder in self.encoders.items():
            filename = f"bpo_{encoder_name}_{timestamp}.joblib"
            joblib.dump(encoder, filename)

    def generate_training_report(self, approval_results, rejection_results, probability_results):
        """Generate comprehensive training report"""
        print("\n" + "="*80)
        print("🤖 BPO CLAIMS ML TRAINING REPORT")
        print("="*80)

        # Approval Prediction Results
        print(f"\n📈 APPROVAL PREDICTION MODEL")
        print(f"Model Type: {approval_results['model_type']}")
        print(f"Accuracy: {approval_results['accuracy']:.1%}")
        print(f"\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(list(approval_results['feature_importance'].items())[:5], 1):
            print(f"   {i}. {feature}: {importance:.3f}")

        # Rejection Reason Prediction Results
        if rejection_results:
            print(f"\n🚫 REJECTION REASON PREDICTION MODEL")
            print(f"Accuracy: {rejection_results['accuracy']:.1%}")
            print(f"Number of rejection categories: {len(rejection_results['class_names'])}")

        # Probability Prediction Results
        print(f"\n📊 PROBABILITY PREDICTION MODEL")
        print(f"Mean Absolute Error: {probability_results['mae']:.4f}")
        print(f"Average prediction error: ±{probability_results['mae']*100:.1f} percentage points")

        # Business Impact Estimation
        print(f"\n💰 ESTIMATED BUSINESS IMPACT")
        current_approval_rate = 0.85  # Assume current 85% approval rate
        improved_approval_rate = min(0.95, current_approval_rate + (approval_results['accuracy'] - 0.5) * 0.2)
        improvement = improved_approval_rate - current_approval_rate

        print(f"Current approval rate: {current_approval_rate:.1%}")
        print(f"Projected approval rate: {improved_approval_rate:.1%}")
        print(f"Expected improvement: +{improvement:.1%}")
        print(f"Revenue impact: {improvement*100:.1f}% increase in collected revenue")

        # Implementation Recommendations
        print(f"\n🎯 IMPLEMENTATION RECOMMENDATIONS")
        recommendations = [
            "Deploy approval classifier for real-time claim screening",
            "Use probability scores to prioritize high-confidence claims",
            "Implement rejection reason prediction for proactive issue resolution",
            "Set approval probability threshold at 0.75 for submission decisions",
            "Monitor model performance and retrain monthly with new data"
        ]

        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

def main():
    """Main training pipeline execution"""
    print("🚀 Starting BPO Claims ML Training Pipeline")

    trainer = BPOMLTrainer()
    results = trainer.train_all_models()

    print(f"\n✅ Training completed successfully!")
    print(f"Models saved with timestamp for production deployment")

if __name__ == "__main__":
    main()