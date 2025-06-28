# priority_ml.py - Priority Prediction and Predictive Maintenance ML Engine
"""
Specialized ML engine for TAQA equipment data:
1. Priority Prediction: Train on first 4000 rows to predict missing priorities
2. Predictive Maintenance: Forecast equipment failure based on historical patterns
3. Data Cleaning: Remove duplicates and prepare data for ML
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from datetime import datetime, timedelta
import joblib
import logging
import re
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class TaqaPriorityML:
    """
    TAQA Priority Prediction and Predictive Maintenance ML Engine
    
    Features:
    - Priority prediction for missing values
    - Equipment failure prediction
    - Data cleaning and preprocessing
    - Text analysis of descriptions
    """
    
    def __init__(self):
        self.priority_model = None
        self.failure_model = None
        self.text_vectorizer = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Equipment failure patterns (keywords that indicate problems)
        self.failure_keywords = [
            'fuite', 'leak', 'défaillance', 'failure', 'panne', 'breakdown',
            'vibration', 'noise', 'bruit', 'surchauffe', 'overheating',
            'étanche', 'seal', 'usure', 'wear', 'casse', 'broken',
            'colmatage', 'blockage', 'corrosion', 'erosion'
        ]
        
        # Priority mapping based on severity
        self.priority_mapping = {
            'critique': 1, 'critical': 1, 'urgent': 1,
            'élevé': 2, 'high': 2, 'important': 2,
            'moyen': 3, 'medium': 3, 'normal': 3,
            'faible': 4, 'low': 4, 'mineur': 4,
            'très faible': 5, 'very low': 5, 'cosmetic': 5
        }

    def load_and_clean_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data and perform cleaning operations
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"📊 Loading data from {file_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            logger.info(f"✅ Data loaded successfully. Shape: {df.shape}")
            
            # Display basic info
            logger.info(f"📋 Columns: {list(df.columns)}")
            logger.info(f"📋 Missing values per column:")
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                logger.info(f"   {col}: {missing_count} missing ({missing_count/len(df)*100:.1f}%)")
            
            # Remove duplicates
            original_count = len(df)
            df = df.drop_duplicates()
            duplicates_removed = original_count - len(df)
            logger.info(f"🧹 Removed {duplicates_removed} duplicate rows")
            
            # Clean and standardize data
            df = self._clean_data_columns(df)
            
            # Analyze priority distribution
            if 'Priorité' in df.columns:
                priority_counts = df['Priorité'].value_counts().sort_index()
                logger.info(f"📊 Priority distribution:")
                for priority, count in priority_counts.items():
                    logger.info(f"   Priority {priority}: {count} rows")
            
            logger.info(f"✅ Data cleaning completed. Final shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def _clean_data_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data columns"""
        
        # Standardize column names (remove spaces, special characters)
        df.columns = df.columns.str.strip()
        
        # Clean text fields
        text_columns = ['Description', 'Description equipement', 'Section proprietaire']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        
        # Clean and parse dates
        if 'Date de detection de l\'anomalie' in df.columns:
            df['Date de detection de l\'anomalie'] = pd.to_datetime(
                df['Date de detection de l\'anomalie'], 
                errors='coerce'
            )
            
            # Extract additional date features
            df['year'] = df['Date de detection de l\'anomalie'].dt.year
            df['month'] = df['Date de detection de l\'anomalie'].dt.month
            df['day_of_week'] = df['Date de detection de l\'anomalie'].dt.dayofweek
            df['quarter'] = df['Date de detection de l\'anomalie'].dt.quarter
        
        # Clean status field
        if 'Statut' in df.columns:
            df['Statut'] = df['Statut'].fillna('Unknown').astype(str).str.strip()
        
        # Clean equipment ID
        if 'Num_equipement' in df.columns:
            df['Num_equipement'] = df['Num_equipement'].astype(str).str.strip()
        
        return df

    def prepare_features_for_priority_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for priority prediction using simple, consistent features
        
        Args:
            df: DataFrame with equipment data
            
        Returns:
            Feature matrix
        """
        logger.info("🔧 Preparing features for priority prediction...")
        
        features = []
        
        # Simple text-based features (consistent across calls)
        if 'Description' in df.columns:
            descriptions = df['Description'].fillna('').astype(str)
            
            # Count-based features (always same dimension)
            text_features = []
            for desc in descriptions:
                desc_lower = desc.lower()
                # Feature: description length
                text_features.append([
                    len(desc),  # Description length
                    desc.count(' '),  # Word count
                    len([w for w in self.failure_keywords if w in desc_lower]),  # Failure keyword count
                    1 if any(urgent in desc_lower for urgent in ['urgent', 'critique', 'immédiat']) else 0,  # Urgency flag
                    1 if 'fuite' in desc_lower else 0,  # Leak indicator
                    1 if 'vibration' in desc_lower else 0,  # Vibration indicator
                    1 if any(temp in desc_lower for temp in ['chaud', 'surchauffe', 'température']) else 0,  # Temperature indicator
                    1 if any(maint in desc_lower for maint in ['maintenance', 'révision', 'contrôle']) else 0,  # Maintenance indicator
                ])
            
            features.append(np.array(text_features))
        
        # Equipment type features
        if 'Description equipement' in df.columns:
            if 'equipment_type' not in self.label_encoders:
                self.label_encoders['equipment_type'] = LabelEncoder()
                equipment_encoded = self.label_encoders['equipment_type'].fit_transform(df['Description equipement'])
            else:
                # Handle unseen labels by assigning them to a default value
                equipment_values = df['Description equipement'].copy()
                known_labels = set(self.label_encoders['equipment_type'].classes_)
                equipment_values = equipment_values.apply(
                    lambda x: x if x in known_labels else self.label_encoders['equipment_type'].classes_[0]
                )
                equipment_encoded = self.label_encoders['equipment_type'].transform(equipment_values)
            
            features.append(equipment_encoded.reshape(-1, 1))
        
        # Section features
        if 'Section proprietaire' in df.columns:
            if 'section' not in self.label_encoders:
                self.label_encoders['section'] = LabelEncoder()
                section_encoded = self.label_encoders['section'].fit_transform(df['Section proprietaire'])
            else:
                # Handle unseen labels
                section_values = df['Section proprietaire'].copy()
                known_labels = set(self.label_encoders['section'].classes_)
                section_values = section_values.apply(
                    lambda x: x if x in known_labels else self.label_encoders['section'].classes_[0]
                )
                section_encoded = self.label_encoders['section'].transform(section_values)
            
            features.append(section_encoded.reshape(-1, 1))
        
        # Status features
        if 'Statut' in df.columns:
            if 'status' not in self.label_encoders:
                self.label_encoders['status'] = LabelEncoder()
                status_encoded = self.label_encoders['status'].fit_transform(df['Statut'])
            else:
                # Handle unseen labels
                status_values = df['Statut'].copy()
                known_labels = set(self.label_encoders['status'].classes_)
                status_values = status_values.apply(
                    lambda x: x if x in known_labels else self.label_encoders['status'].classes_[0]
                )
                status_encoded = self.label_encoders['status'].transform(status_values)
            
            features.append(status_encoded.reshape(-1, 1))
        
        # Date features (always include, use defaults for missing dates)
        date_features = []
        if 'Date de detection de l\'anomalie' in df.columns:
            for col in ['year', 'month', 'day_of_week', 'quarter']:
                if col in df.columns:
                    date_features.append(df[col].fillna(2019).values)  # Default to 2019 if missing
                else:
                    # Create default values if column doesn't exist
                    if col == 'year':
                        date_features.append(np.full(len(df), 2019))
                    elif col == 'month':
                        date_features.append(np.full(len(df), 1))
                    elif col == 'day_of_week':
                        date_features.append(np.full(len(df), 0))
                    elif col == 'quarter':
                        date_features.append(np.full(len(df), 1))
        else:
            # No date column at all, create default date features
            date_features = [
                np.full(len(df), 2019),  # year
                np.full(len(df), 1),     # month
                np.full(len(df), 0),     # day_of_week
                np.full(len(df), 1)      # quarter
            ]
        
        if date_features:
            features.append(np.column_stack(date_features))
        
        # Failure pattern features (based on description keywords) - always include
        if 'Description' in df.columns:
            failure_patterns = self._extract_failure_patterns(df['Description'])
        else:
            failure_patterns = np.zeros(len(df))
        
        features.append(failure_patterns.reshape(-1, 1))
        
        # Combine all features
        if features:
            X = np.hstack(features)
        else:
            raise ValueError("No features could be extracted from the data")
        
        logger.info(f"✅ Features prepared. Shape: {X.shape}")
        return X

    def _extract_failure_patterns(self, descriptions: pd.Series) -> np.ndarray:
        """Extract failure pattern indicators from descriptions"""
        
        failure_scores = []
        for desc in descriptions:
            desc_lower = str(desc).lower()
            score = 0
            
            # Count failure keywords
            for keyword in self.failure_keywords:
                if keyword in desc_lower:
                    score += 1
            
            # Look for urgency indicators
            urgency_words = ['urgent', 'immédiat', 'critique', 'important']
            for word in urgency_words:
                if word in desc_lower:
                    score += 2
            
            failure_scores.append(score)
        
        return np.array(failure_scores)

    def train_priority_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the priority prediction model using first 4000 rows
        
        Args:
            df: Complete DataFrame
            
        Returns:
            Training results and metrics
        """
        logger.info("🤖 Training priority prediction model...")
        
        # Get rows with priority data (first 4000)
        df_with_priority = df[df['Priorité'].notna()].head(4000)
        
        if len(df_with_priority) == 0:
            raise ValueError("No rows with priority data found")
        
        logger.info(f"📊 Training on {len(df_with_priority)} rows with priority data")
        
        # Prepare features and labels
        X = self.prepare_features_for_priority_prediction(df_with_priority)
        y = df_with_priority['Priorité'].values
        
        # Split for training and validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.priority_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.priority_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.priority_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get feature importance
        feature_names = self._get_feature_names()
        feature_importance = dict(zip(feature_names, self.priority_model.feature_importances_))
        
        results = {
            "model_type": "RandomForestClassifier",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": feature_importance,
            "priority_distribution": df_with_priority['Priorité'].value_counts().to_dict()
        }
        
        logger.info(f"✅ Priority model trained successfully. Accuracy: {accuracy:.3f}")
        return results

    def predict_missing_priorities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict priorities for rows with missing priority values
        
        Args:
            df: DataFrame with some missing priorities
            
        Returns:
            DataFrame with predicted priorities filled in
        """
        if self.priority_model is None:
            raise ValueError("Priority model not trained. Call train_priority_model first.")
        
        logger.info("🔮 Predicting missing priorities...")
        
        # Get rows with missing priorities
        missing_priority_mask = df['Priorité'].isna()
        df_missing = df[missing_priority_mask].copy()
        
        if len(df_missing) == 0:
            logger.info("✅ No missing priorities found")
            return df
        
        logger.info(f"📊 Predicting priorities for {len(df_missing)} rows")
        
        # Prepare features for missing priority rows
        X_missing = self.prepare_features_for_priority_prediction(df_missing)
        X_missing_scaled = self.scaler.transform(X_missing)
        
        # Predict priorities
        predicted_priorities = self.priority_model.predict(X_missing_scaled)
        prediction_probabilities = self.priority_model.predict_proba(X_missing_scaled)
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(prediction_probabilities, axis=1)
        
        # Fill in the predictions
        df_result = df.copy()
        df_result.loc[missing_priority_mask, 'Priorité'] = predicted_priorities
        df_result.loc[missing_priority_mask, 'predicted_priority_confidence'] = confidence_scores
        
        # Log prediction summary
        prediction_counts = pd.Series(predicted_priorities).value_counts().sort_index()
        logger.info(f"📊 Predicted priority distribution:")
        for priority, count in prediction_counts.items():
            logger.info(f"   Priority {priority}: {count} predictions")
        
        avg_confidence = np.mean(confidence_scores)
        logger.info(f"🎯 Average prediction confidence: {avg_confidence:.3f}")
        
        return df_result

    def train_failure_prediction_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train predictive maintenance model to forecast equipment failures
        
        Args:
            df: DataFrame with historical equipment data
            
        Returns:
            Training results and metrics
        """
        logger.info("🔧 Training failure prediction model...")
        
        # Prepare data for failure prediction
        df_failure = self._prepare_failure_prediction_data(df)
        
        if len(df_failure) < 100:
            raise ValueError("Insufficient data for failure prediction training")
        
        # Prepare features
        X = self.prepare_features_for_priority_prediction(df_failure)
        y = df_failure['days_to_next_failure'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regression model for days to failure
        self.failure_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        self.failure_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.failure_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        results = {
            "model_type": "RandomForestRegressor",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "rmse": rmse,
            "mean_actual_days": np.mean(y_test),
            "mean_predicted_days": np.mean(y_pred)
        }
        
        self.is_trained = True
        logger.info(f"✅ Failure prediction model trained. RMSE: {rmse:.2f} days")
        return results

    def _prepare_failure_prediction_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for failure prediction by calculating time between failures"""
        
        # Sort by equipment and date
        df_sorted = df.sort_values(['Num_equipement', 'Date de detection de l\'anomalie'])
        
        # Calculate days to next failure for each equipment
        df_failure = []
        
        for equipment_id in df_sorted['Num_equipement'].unique():
            equipment_data = df_sorted[df_sorted['Num_equipement'] == equipment_id].copy()
            
            if len(equipment_data) < 2:
                continue
            
            # Calculate days between consecutive failures
            equipment_data['next_failure_date'] = equipment_data['Date de detection de l\'anomalie'].shift(-1)
            equipment_data['days_to_next_failure'] = (
                equipment_data['next_failure_date'] - equipment_data['Date de detection de l\'anomalie']
            ).dt.days
            
            # Remove last row (no next failure) and invalid data
            equipment_data = equipment_data[:-1]
            equipment_data = equipment_data[equipment_data['days_to_next_failure'] > 0]
            equipment_data = equipment_data[equipment_data['days_to_next_failure'] <= 365]  # Max 1 year
            
            if len(equipment_data) > 0:
                df_failure.append(equipment_data)
        
        if df_failure:
            result = pd.concat(df_failure, ignore_index=True)
            logger.info(f"📊 Prepared {len(result)} records for failure prediction")
            return result
        else:
            return pd.DataFrame()

    def predict_equipment_failure_original(self, equipment_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict when a specific equipment might fail next
        
        Args:
            equipment_id: ID of the equipment
            df: Historical data DataFrame
            
        Returns:
            Failure prediction results
        """
        if self.failure_model is None:
            raise ValueError("Failure model not trained. Call train_failure_prediction_model first.")
        
        # Get latest data for this equipment
        equipment_data = df[df['Num_equipement'] == equipment_id].copy()
        
        if len(equipment_data) == 0:
            raise ValueError(f"No data found for equipment {equipment_id}")
        
        # Get most recent record
        latest_record = equipment_data.sort_values('Date de detection de l\'anomalie').iloc[-1:].copy()
        
        # Prepare features
        X = self.prepare_features_for_priority_prediction(latest_record)
        X_scaled = self.scaler.transform(X)
        
        # Predict days to failure
        predicted_days = self.failure_model.predict(X_scaled)[0]
        
        # Calculate predicted failure date
        last_failure_date = latest_record['Date de detection de l\'anomalie'].iloc[0]
        predicted_failure_date = last_failure_date + timedelta(days=int(predicted_days))
        
        # Calculate risk level
        if predicted_days <= 7:
            risk_level = "CRITICAL"
            risk_color = "#dc2626"
        elif predicted_days <= 30:
            risk_level = "HIGH"
            risk_color = "#ea580c"
        elif predicted_days <= 90:
            risk_level = "MEDIUM"
            risk_color = "#d97706"
        else:
            risk_level = "LOW"
            risk_color = "#059669"
        
        # Get equipment history
        failure_count = len(equipment_data)
        avg_days_between_failures = 0
        if failure_count > 1:
            equipment_data_sorted = equipment_data.sort_values('Date de detection de l\'anomalie')
            date_diffs = equipment_data_sorted['Date de detection de l\'anomalie'].diff().dt.days
            avg_days_between_failures = date_diffs.mean()
        
        return {
            "equipment_id": equipment_id,
            "predicted_days_to_failure": round(predicted_days, 1),
            "predicted_failure_date": predicted_failure_date.isoformat(),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence": "medium",  # Could be enhanced with prediction intervals
            "last_failure_date": last_failure_date.isoformat(),
            "historical_failure_count": failure_count,
            "avg_days_between_failures": round(avg_days_between_failures, 1) if avg_days_between_failures > 0 else None,
            "equipment_type": latest_record['Description equipement'].iloc[0],
            "section": latest_record['Section proprietaire'].iloc[0]
        }

    def _get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        names = []
        
        # Text-based features (fixed set)
        names.extend([
            'desc_length', 'word_count', 'failure_keywords', 'urgency_flag',
            'leak_indicator', 'vibration_indicator', 'temperature_indicator', 'maintenance_indicator'
        ])
        
        names.extend(['equipment_type', 'section', 'status'])
        names.extend(['year', 'month', 'day_of_week', 'quarter'])
        names.append('failure_pattern_score')
        
        return names[:len(self.priority_model.feature_importances_)] if self.priority_model else names

    def save_models(self, model_dir: str = "models"):
        """Save trained models to disk"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.priority_model:
            joblib.dump(self.priority_model, f"{model_dir}/priority_model.pkl")
            
        if self.failure_model:
            joblib.dump(self.failure_model, f"{model_dir}/failure_model.pkl")
            
        if self.text_vectorizer:
            joblib.dump(self.text_vectorizer, f"{model_dir}/text_vectorizer.pkl")
            
        joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.pkl")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        
        logger.info(f"✅ Models saved to {model_dir}/")

    def load_models(self, model_dir: str = "models"):
        """Load trained models from disk"""
        import os
        
        try:
            if os.path.exists(f"{model_dir}/priority_model.pkl"):
                self.priority_model = joblib.load(f"{model_dir}/priority_model.pkl")
                
            if os.path.exists(f"{model_dir}/failure_model.pkl"):
                self.failure_model = joblib.load(f"{model_dir}/failure_model.pkl")
                
            if os.path.exists(f"{model_dir}/text_vectorizer.pkl"):
                self.text_vectorizer = joblib.load(f"{model_dir}/text_vectorizer.pkl")
                
            if os.path.exists(f"{model_dir}/label_encoders.pkl"):
                self.label_encoders = joblib.load(f"{model_dir}/label_encoders.pkl")
                
            if os.path.exists(f"{model_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{model_dir}/scaler.pkl")
                
            self.is_trained = True
            logger.info(f"✅ Models loaded from {model_dir}/")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            return False

    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train both priority prediction and failure prediction models
        
        Args:
            df: Training data DataFrame
            
        Returns:
            Training results
        """
        try:
            # Clean data first
            df_clean = self._clean_data_columns(df)
            
            # Train priority model
            priority_results = self.train_priority_model(df_clean)
            
            # Try to train failure model, but don't fail if insufficient data
            failure_results = None
            try:
                failure_results = self.train_failure_prediction_model(df_clean)
                failure_trained = True
            except ValueError as e:
                if "Insufficient data" in str(e):
                    logger.warning("⚠️ Insufficient data for failure prediction training, skipping...")
                    failure_results = {
                        "status": "skipped",
                        "reason": "insufficient_data",
                        "message": "Need more historical failure data for predictive maintenance"
                    }
                    failure_trained = False
                else:
                    raise e
            
            self.is_trained = True  # At least priority model is trained
            
            return {
                "status": "success",
                "priority_model": priority_results,
                "failure_model": failure_results,
                "training_samples": len(df_clean),
                "duplicates_removed": len(df) - len(df_clean),
                "models_trained": ["priority_classifier"] + (["failure_regressor"] if failure_trained else []),
                "notes": [] if failure_trained else ["Failure prediction model not trained due to insufficient data"]
            }
            
        except Exception as e:
            logger.error(f"❌ Error training models: {e}")
            raise e

    def predict_equipment_failure(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Predict failure for all equipment in the dataframe
        
        Args:
            df: DataFrame with equipment data
            
        Returns:
            List of failure predictions for all equipment
        """
        if self.failure_model is None:
            raise ValueError("Failure model not trained. Call train_models first.")
        
        try:
            # Clean data
            df_clean = self._clean_data_columns(df)
            
            # Get unique equipment IDs
            if 'Num_equipement' not in df_clean.columns:
                logger.warning("No 'Num_equipement' column found, creating generic equipment IDs")
                df_clean['Num_equipement'] = df_clean.index.astype(str)
            
            equipment_ids = df_clean['Num_equipement'].unique()
            
            results = []
            for equipment_id in equipment_ids:
                try:
                    # Get failure prediction for this equipment
                    result = self.predict_equipment_failure_single(equipment_id, df_clean)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Could not predict for equipment {equipment_id}: {e}")
                    # Add a basic result for failed predictions
                    results.append({
                        "equipment_id": equipment_id,
                        "error": str(e),
                        "risk_level": "UNKNOWN",
                        "predicted_days_to_failure": None
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in batch equipment failure prediction: {e}")
            raise e

    def predict_equipment_failure_single(self, equipment_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict when a specific equipment might fail next (renamed from original method)
        
        Args:
            equipment_id: ID of the equipment
            df: Historical data DataFrame
            
        Returns:
            Failure prediction results
        """
        if self.failure_model is None:
            raise ValueError("Failure model not trained. Call train_models first.")
        
        # Get latest data for this equipment
        equipment_data = df[df['Num_equipement'] == equipment_id].copy()
        
        if len(equipment_data) == 0:
            raise ValueError(f"No data found for equipment {equipment_id}")
        
        # Get most recent record
        if 'Date de detection de l\'anomalie' in equipment_data.columns:
            latest_record = equipment_data.sort_values('Date de detection de l\'anomalie').iloc[-1:].copy()
            last_failure_date = latest_record['Date de detection de l\'anomalie'].iloc[0]
        else:
            latest_record = equipment_data.iloc[-1:].copy()
            last_failure_date = datetime.now()
        
        # Prepare features
        X = self.prepare_features_for_priority_prediction(latest_record)
        X_scaled = self.scaler.transform(X)
        
        # Predict days to failure
        predicted_days = self.failure_model.predict(X_scaled)[0]
        
        # Calculate predicted failure date
        predicted_failure_date = last_failure_date + timedelta(days=int(predicted_days))
        
        # Calculate risk level and failure probability
        failure_probability = max(0, min(1, 1 - (predicted_days / 365)))  # Higher probability for sooner failures
        
        if predicted_days <= 7:
            risk_level = "CRITICAL"
            risk_color = "#dc2626"
        elif predicted_days <= 30:
            risk_level = "HIGH"
            risk_color = "#ea580c"
        elif predicted_days <= 90:
            risk_level = "MEDIUM"
            risk_color = "#d97706"
        else:
            risk_level = "LOW"
            risk_color = "#059669"
        
        # Get equipment history
        failure_count = len(equipment_data)
        avg_days_between_failures = 0
        if failure_count > 1 and 'Date de detection de l\'anomalie' in equipment_data.columns:
            equipment_data_sorted = equipment_data.sort_values('Date de detection de l\'anomalie')
            date_diffs = equipment_data_sorted['Date de detection de l\'anomalie'].diff().dt.days
            avg_days_between_failures = date_diffs.mean()
        
        return {
            "equipment_id": equipment_id,
            "predicted_days_to_failure": round(predicted_days, 1),
            "predicted_failure_date": predicted_failure_date.isoformat(),
            "failure_probability": round(failure_probability, 3),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "confidence": "medium",
            "last_failure_date": last_failure_date.isoformat() if isinstance(last_failure_date, datetime) else str(last_failure_date),
            "historical_failure_count": failure_count,
            "avg_days_between_failures": round(avg_days_between_failures, 1) if avg_days_between_failures > 0 else None,
            "equipment_type": latest_record.get('Description equipement', ['Unknown']).iloc[0] if 'Description equipement' in latest_record.columns else "Unknown",
            "section": latest_record.get('Section proprietaire', ['Unknown']).iloc[0] if 'Section proprietaire' in latest_record.columns else "Unknown"
        }

    def get_equipment_forecast(self, equipment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get failure forecast for specific equipment (requires historical data to be loaded)
        
        Args:
            equipment_id: ID of the equipment
            
        Returns:
            Equipment forecast or None if not found
        """
        # This method would need historical data stored in the class
        # For now, return a placeholder that indicates data is needed
        return {
            "equipment_id": equipment_id,
            "message": "Historical data required for forecast",
            "status": "data_needed",
            "note": "Use /priority_ml/predictive_maintenance endpoint with data file for forecasting"
        }

    def get_maintenance_recommendations(self, forecast: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Get maintenance recommendations based on failure forecast
        
        Args:
            forecast: Failure forecast results
            
        Returns:
            List of maintenance recommendations
        """
        recommendations = []
        
        if not forecast or forecast.get('status') == 'data_needed':
            return [{"type": "info", "message": "Upload equipment data to get specific recommendations"}]
        
        predicted_days = forecast.get('predicted_days_to_failure', 365)
        risk_level = forecast.get('risk_level', 'LOW')
        failure_probability = forecast.get('failure_probability', 0)
        
        if risk_level == "CRITICAL":
            recommendations.extend([
                {"type": "urgent", "message": "IMMEDIATE ACTION REQUIRED: Schedule emergency maintenance within 24 hours"},
                {"type": "safety", "message": "Consider taking equipment offline to prevent safety incidents"},
                {"type": "resources", "message": "Mobilize maintenance team and ensure spare parts availability"}
            ])
        elif risk_level == "HIGH":
            recommendations.extend([
                {"type": "priority", "message": "Schedule maintenance within 1 week"},
                {"type": "monitoring", "message": "Increase monitoring frequency and conduct daily inspections"},
                {"type": "preparation", "message": "Order spare parts and schedule maintenance window"}
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                {"type": "planning", "message": "Schedule maintenance within 1 month"},
                {"type": "monitoring", "message": "Conduct weekly inspections and monitor key parameters"},
                {"type": "inventory", "message": "Check spare parts inventory and order if needed"}
            ])
        else:
            recommendations.extend([
                {"type": "routine", "message": "Continue routine maintenance schedule"},
                {"type": "monitoring", "message": "Monthly condition monitoring is sufficient"},
                {"type": "optimization", "message": "Consider optimizing operating parameters for efficiency"}
            ])
        
        # Add specific recommendations based on failure probability
        if failure_probability > 0.8:
            recommendations.append({"type": "analysis", "message": "Conduct root cause analysis to identify failure patterns"})
        
        if failure_probability > 0.5:
            recommendations.append({"type": "backup", "message": "Identify backup equipment or alternative processes"})
        
        return recommendations
