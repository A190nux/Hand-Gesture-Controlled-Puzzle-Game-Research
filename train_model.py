import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import json
import pickle
import os
import sys
from importlib.metadata import distributions
import subprocess
import time
import requests

class HandGestureTrainer:
    def __init__(self, models_dir="Models", mlflow_port=5000, auto_start_mlflow=True):
        """Initialize the Hand Gesture Trainer with XGBoost"""
        self.models_dir = models_dir
        self.mlflow_port = mlflow_port
        self.mlflow_uri = f"http://localhost:{mlflow_port}"
        self.mlflow_process = None
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs("mlflow_data", exist_ok=True)
        os.makedirs("mlflow_artifacts", exist_ok=True)
        
        if auto_start_mlflow:
            self._start_mlflow_server()
        
        # Setup MLflow connection
        mlflow.set_tracking_uri(self.mlflow_uri)
        print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")
        
        # Setup experiment
        experiment_name = "hand_gesture_xgboost"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(f"✓ Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Using existing experiment: {experiment_name}")
        except Exception as e:
            print(f"❌ MLflow connection failed: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        print(f"✓ Experiment ID: {experiment_id}")
    
    def _start_mlflow_server(self):
        """Start MLflow server if not already running"""
        # Check if server is already running
        if self._is_mlflow_running():
            print(f"✓ MLflow server already running on port {self.mlflow_port}")
            return
        
        print(f"🚀 Starting MLflow server on port {self.mlflow_port}...")
        
        try:
            # Start MLflow server in background
            self.mlflow_process = subprocess.Popen([
                "mlflow", "server",
                "--backend-store-uri", "./mlflow_data",
                "--default-artifact-root", "./mlflow_artifacts",
                "--host", "127.0.0.1",
                "--port", str(self.mlflow_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            max_wait = 30  # seconds
            wait_time = 0
            while wait_time < max_wait:
                if self._is_mlflow_running():
                    print(f"✅ MLflow server started successfully!")
                    print(f"🌐 MLflow UI available at: {self.mlflow_uri}")
                    return
                time.sleep(1)
                wait_time += 1
            
            raise Exception(f"MLflow server failed to start within {max_wait} seconds")
            
        except Exception as e:
            print(f"❌ Failed to start MLflow server: {e}")
            raise
    
    def _is_mlflow_running(self):
        """Check if MLflow server is running"""
        try:
            response = requests.get(f"{self.mlflow_uri}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def cleanup(self):
        """Clean up MLflow server process"""
        if self.mlflow_process:
            print("🛑 Stopping MLflow server...")
            self.mlflow_process.terminate()
            self.mlflow_process.wait()
            print("✓ MLflow server stopped")

    def log_environment_info(self):
        """Log environment information to MLflow"""
        try:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            mlflow.log_param("python_version", python_version)
            mlflow.log_param("platform", sys.platform)
            
            # Log key package versions
            installed_packages = {dist.metadata['name']: dist.version for dist in distributions()}
            key_packages = ["numpy", "pandas", "xgboost", "scikit-learn", "mlflow"]
            
            for package in key_packages:
                if package in installed_packages:
                    mlflow.log_param(f"version_{package.replace('-', '_')}", installed_packages[package])
        except Exception as e:
            print(f"⚠️  Warning: Could not log environment info: {e}")
    
    def load_data(self, csv_path):
        """Load data from CSV file following your exact steps"""
        print(f"📂 Loading data from: {csv_path}")
        
        # Load CSV exactly as in your notebook
        df = pd.read_csv(csv_path)
        
        print(f"✓ Data loaded: {df.shape[0]} samples, {df.shape[1]} columns")
        
        # Split features and labels exactly as in your notebook
        X = df.drop('label', axis=1)
        y = df['label']
        
        print(f"✓ Features shape: {X.shape}")
        print(f"✓ Labels shape: {y.shape}")
        print(f"✓ Unique labels: {sorted(y.unique())}")
        
        return X, y
    
    def custom_scaling(self, df):
        """
        Normalize the data by centering using the wrist and scaling using the middle finger tip
        Exactly as in your notebook
        """
        x_cols = [f'x{i}' for i in range(1, 22)]
        y_cols = [f'y{i}' for i in range(1, 22)]
        
        x_translate = df['x1'].copy()
        y_translate = df['y1'].copy()
        x_scale = df['x13'].copy() - x_translate
        y_scale = df['y13'].copy() - y_translate
        
        # Use WRIST (x1,y1) as origin
        for col in x_cols:
            df[col] = df[col] - x_translate
        for col in y_cols:
            df[col] = df[col] - y_translate
        
        # Use MID_FINGER_TIP (x13,y13) for scaling
        for col in x_cols:
            df[col] = df[col] / x_scale  # Mid finger tip is column 13 (1-based)
        for col in y_cols:
            df[col] = df[col] / y_scale
        
        # Leave z-values untouched
        return df
    
    def train_model(self, X_train_scaled, y_train_encoded, use_best_params=True, **custom_params):
        """Train XGBoost model with your exact parameters"""
        print("🏋️  Training XGBoost model...")
        
        if use_best_params:
            # Use the best parameters from your notebook
            model_params = {
                'colsample_bytree': 1.0,
                'learning_rate': 0.2,
                'max_depth': 7,
                'n_estimators': 200,
                'subsample': 0.8,
                'tree_method': 'hist'
                # 'device': "cuda"  # Uncomment if you want to use GPU
            }
            print("📊 Using best parameters from hyperparameter tuning")
        else:
            # Default parameters
            model_params = {
                'colsample_bytree': 0.8,
                'learning_rate': 0.1,
                'max_depth': 6,
                'n_estimators': 100,
                'subsample': 0.8,
                'tree_method': 'hist'
            }
        
        # Update with any custom parameters
        model_params.update(custom_params)
        
        # Create and train model exactly as in your notebook
        best_xgb_model = XGBClassifier(**model_params)
        best_xgb_model.fit(X_train_scaled, y_train_encoded)
        
        print("✓ Model training completed")
        return best_xgb_model, model_params
    
    def evaluate_model(self, model, X_test_scaled, y_test_encoded, label_encoder):
        """Evaluate model and return metrics"""
        print("📊 Evaluating model...")
        
        # Predictions
        y_pred_encoded = model.predict(X_test_scaled)
        
        # Convert back to original labels for reporting
        y_test_original = label_encoder.inverse_transform(y_test_encoded)
        y_pred_original = label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        
        # Classification report with original class names
        class_names = label_encoder.classes_
        report = classification_report(y_test_original, y_pred_original, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test_encoded, y_pred_encoded)
        
        print(f"✓ Accuracy: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test_original, y_pred_original, target_names=class_names))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions_encoded': y_pred_encoded,
            'predictions_original': y_pred_original,
            'class_names': class_names
        }
    
    def save_model_artifacts(self, model, label_encoder, model_params, run_id):
        """Save model and related artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_filename = f"xgboost_model_{timestamp}.pkl"
        model_path = os.path.join(self.models_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save label encoder
        encoder_filename = f"label_encoder_{timestamp}.pkl"
        encoder_path = os.path.join(self.models_dir, encoder_filename)
        
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        # Save class names as JSON for easy reading
        classes_filename = f"class_names_{timestamp}.json"
        classes_path = os.path.join(self.models_dir, classes_filename)
        
        with open(classes_path, 'w') as f:
            json.dump(label_encoder.classes_.tolist(), f)
        
        # Save model info
        model_info = {
            'model_type': 'XGBoost',
            'model_file': model_filename,
            'encoder_file': encoder_filename,
            'classes_file': classes_filename,
            'num_classes': len(label_encoder.classes_),
            'class_names': label_encoder.classes_.tolist(),
            'model_params': model_params,
            'mlflow_run_id': run_id,
            'timestamp': timestamp,
            'preprocessing': 'custom_scaling_wrist_centered'
        }
        
        info_filename = f"model_info_{timestamp}.json"
        info_path = os.path.join(self.models_dir, info_filename)
        
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"✓ Model saved: {model_path}")
        print(f"✓ Label encoder saved: {encoder_path}")
        print(f"✓ Classes saved: {classes_path}")
        print(f"✓ Model info saved: {info_path}")
        
        return model_path, encoder_path, classes_path, info_path
    
    def create_confusion_matrix_plot(self, confusion_matrix, class_names):
        """Create and save confusion matrix visualization"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = os.path.join(self.models_dir, f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(cm_path)
        plt.close()
        
        return cm_path
    
    def run_experiment(self, csv_path, test_size=0.2, random_state=69, use_best_params=True, **custom_params):
        """Run complete training experiment following your exact notebook steps"""
        
        try:
            with mlflow.start_run() as run:
                print(f"🚀 Starting MLflow run: {run.info.run_id}")
                
                # Log environment info
                self.log_environment_info()
                
                # Step 1: Load data exactly as in your notebook
                X, y = self.load_data(csv_path)
                
                if len(X) == 0:
                    raise ValueError("No data loaded. Check your CSV file path.")
                
                # Log dataset info
                try:
                    mlflow.log_param("csv_path", csv_path)
                    mlflow.log_param("total_samples", len(X))
                    mlflow.log_param("num_classes", len(y.unique()))
                    mlflow.log_param("feature_dim", X.shape[1])
                    mlflow.log_param("test_size", test_size)
                    mlflow.log_param("random_state", random_state)
                    mlflow.log_param("use_best_params", use_best_params)
                except Exception as e:
                    print(f"⚠️  Warning: Could not log some parameters: {e}")
                
                # Step 2: Split the data exactly as in your notebook
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
                )
                
                try:
                    mlflow.log_param("train_samples", len(X_train))
                    mlflow.log_param("test_samples", len(X_test))
                except Exception as e:
                    print(f"⚠️  Warning: Could not log split parameters: {e}")
                
                print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test samples")
                
                # Step 3: Apply custom scaling exactly as in your notebook
                print("🔧 Applying custom scaling (wrist-centered, middle finger scaled)...")
                X_train_scaled = self.custom_scaling(X_train.copy())
                X_test_scaled = self.custom_scaling(X_test.copy())
                
                print("✓ Custom scaling applied")
                
                # Step 4: Encode labels exactly as in your notebook
                print("🏷️  Encoding labels...")
                label_encoder = LabelEncoder()
                y_train_encoded = label_encoder.fit_transform(y_train)
                y_test_encoded = label_encoder.transform(y_test)
                
                print(f"✓ Labels encoded: {label_encoder.classes_}")
                
                # Step 5: Train model with your exact parameters
                model, model_params = self.train_model(X_train_scaled, y_train_encoded, 
                                                     use_best_params, **custom_params)
                
                # Log model parameters
                try:
                    for param, value in model_params.items():
                        mlflow.log_param(f"xgb_{param}", value)
                except Exception as e:
                    print(f"⚠️  Warning: Could not log model parameters: {e}")
                
                # Step 6: Evaluate model
                metrics = self.evaluate_model(model, X_test_scaled, y_test_encoded, label_encoder)
                
                # Log metrics
                try:
                    mlflow.log_metric("accuracy", metrics['accuracy'])
                    
                    # Log per-class metrics
                    for class_name in metrics['class_names']:
                        if class_name in metrics['classification_report']:
                            class_metrics = metrics['classification_report'][class_name]
                            mlflow.log_metric(f"precision_{class_name}", class_metrics['precision'])
                            mlflow.log_metric(f"recall_{class_name}", class_metrics['recall'])
                            mlflow.log_metric(f"f1_{class_name}", class_metrics['f1-score'])
                    
                    # Log macro averages
                    macro_avg = metrics['classification_report']['macro avg']
                    mlflow.log_metric("macro_precision", macro_avg['precision'])
                    mlflow.log_metric("macro_recall", macro_avg['recall'])
                    mlflow.log_metric("macro_f1", macro_avg['f1-score'])
                except Exception as e:
                    print(f"⚠️  Warning: Could not log metrics: {e}")
                
                # Create confusion matrix plot
                cm_path = self.create_confusion_matrix_plot(metrics['confusion_matrix'], metrics['class_names'])
                
                # Log model and artifacts to MLflow
                try:
                    self.log_model_with_signature(model, X_train_scaled)
                    mlflow.log_artifact(cm_path)
                except Exception as e:
                    print(f"⚠️  Warning: Could not log model to MLflow: {e}")
                
                # Save local artifacts
                model_path, encoder_path, classes_path, info_path = self.save_model_artifacts(
                    model, label_encoder, model_params, run.info.run_id
                )
                
                # Log local artifacts
                try:
                    mlflow.log_artifact(model_path)
                    mlflow.log_artifact(encoder_path)
                    mlflow.log_artifact(classes_path)
                    mlflow.log_artifact(info_path)
                except Exception as e:
                    print(f"⚠️  Warning: Could not log artifacts: {e}")
                
                print(f"\n✅ Experiment completed successfully!")
                print(f"📊 Final Accuracy: {metrics['accuracy']:.4f}")
                print(f"🔗 MLflow Run ID: {run.info.run_id}")
                
                return {
                    'model': model,
                    'label_encoder': label_encoder,
                    'metrics': metrics,
                    'class_names': metrics['class_names'],
                    'run_id': run.info.run_id,
                    'model_path': model_path,
                    'encoder_path': encoder_path,
                    'classes_path': classes_path
                }
                
        except Exception as mlflow_error:
            print(f"⚠️  MLflow error: {mlflow_error}")
            print("🔄 Running without MLflow tracking...")
            
            # Run without MLflow if there are issues
            return self.run_experiment_without_mlflow(csv_path, test_size, random_state, use_best_params, **custom_params)
    
    def run_experiment_without_mlflow(self, csv_path, test_size=0.2, random_state=69, use_best_params=True, **custom_params):
        """Fallback method to run experiment without MLflow"""
        print("🚀 Starting training without MLflow tracking...")
        
        # Step 1: Load data
        X, y = self.load_data(csv_path)
        
        if len(X) == 0:
            raise ValueError("No data loaded. Check your CSV file path.")
        
        # Step 2: Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=True
        )
        
        print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Step 3: Apply custom scaling
        print("🔧 Applying custom scaling (wrist-centered, middle finger scaled)...")
        X_train_scaled = self.custom_scaling(X_train.copy())
        X_test_scaled = self.custom_scaling(X_test.copy())
        
        print("✓ Custom scaling applied")
        
        # Step 4: Encode labels
        print("🏷️  Encoding labels...")
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        print(f"✓ Labels encoded: {label_encoder.classes_}")
        
        # Step 5: Train model
        model, model_params = self.train_model(X_train_scaled, y_train_encoded, 
                                             use_best_params, **custom_params)
        
        # Step 6: Evaluate model
        metrics = self.evaluate_model(model, X_test_scaled, y_test_encoded, label_encoder)
        
        # Create confusion matrix plot
        cm_path = self.create_confusion_matrix_plot(metrics['confusion_matrix'], metrics['class_names'])
        
        # Save local artifacts
        model_path, encoder_path, classes_path, info_path = self.save_model_artifacts(
            model, label_encoder, model_params, "no_mlflow_run"
        )
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"📊 Final Accuracy: {metrics['accuracy']:.4f}")
        
        return {
            'model': model,
            'label_encoder': label_encoder,
            'metrics': metrics,
            'class_names': metrics['class_names'],
            'run_id': "no_mlflow_run",
            'model_path': model_path,
            'encoder_path': encoder_path,
            'classes_path': classes_path
        }
    
    def log_model_with_signature(self, model, X_sample):
        """Log model with proper signature to MLflow"""
        try:
            from mlflow.models.signature import infer_signature
            import numpy as np
            
            # Create a small sample for signature inference
            sample_input = X_sample.head(5) if hasattr(X_sample, 'head') else X_sample[:5]
            sample_predictions = model.predict(sample_input)
            
            signature = infer_signature(sample_input, sample_predictions)
            
            mlflow.xgboost.log_model(
                model, 
                "model", 
                signature=signature,
                input_example=sample_input.iloc[0:1] if hasattr(sample_input, 'iloc') else sample_input[0:1]
            )
            print("✓ Model logged with signature")
        except Exception as e:
            print(f"⚠️  Warning: Could not log model with signature: {e}")
            mlflow.xgboost.log_model(model, "model")


def main():
    parser = argparse.ArgumentParser(description='Train Hand Gesture Recognition Model with XGBoost')
    parser.add_argument('--csv', default='hand_landmarks_data.csv', 
                       help='Path to CSV file with hand landmarks data')
    parser.add_argument('--no-mlflow', action='store_true',
                       help='Skip MLflow server startup and tracking')
    parser.add_argument('--mlflow-port', type=int, default=5000,
                       help='MLflow server port (default: 5000)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=69,
                       help='Random state for reproducibility (default: 69)')
    parser.add_argument('--use-default-params', action='store_true',
                       help='Use default parameters instead of best tuned parameters')
    parser.add_argument('--keep-mlflow', action='store_true',
                       help='Keep MLflow server running after training completes')


    
    # Custom XGBoost parameters (override best params if provided)
    parser.add_argument('--max-depth', type=int, 
                       help='Maximum depth of trees (default: 7 for best params)')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (default: 0.2 for best params)')
    parser.add_argument('--n-estimators', type=int,
                       help='Number of estimators (default: 200 for best params)')
    parser.add_argument('--colsample-bytree', type=float,
                       help='Subsample ratio of columns (default: 1.0 for best params)')
    parser.add_argument('--subsample', type=float,
                       help='Subsample ratio of training instances (default: 0.8 for best params)')
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"❌ Error: CSV file '{args.csv}' not found!")
        print("Make sure your hand_landmarks_data.csv file is in the current directory.")
        return
    
    # Initialize trainer
    trainer = None
    try:
        # Initialize trainer
        if args.no_mlflow:
            print("⚠️  Running without MLflow tracking")
            trainer = HandGestureTrainer(auto_start_mlflow=False)
        else:
            trainer = HandGestureTrainer(mlflow_port=args.mlflow_port, auto_start_mlflow=True)
    except Exception as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        print("Please start MLflow server first:")
        print("mlflow server --backend-store-uri ./mlflow_data --default-artifact-root ./mlflow_artifacts --host 127.0.0.1 --port 5000")
        return
    
    # Determine whether to use best params or default
    use_best_params = not args.use_default_params
    
    # Custom parameters (only include if specified)
    custom_params = {}
    if args.max_depth is not None:
        custom_params['max_depth'] = args.max_depth
    if args.learning_rate is not None:
        custom_params['learning_rate'] = args.learning_rate
    if args.n_estimators is not None:
        custom_params['n_estimators'] = args.n_estimators
    if args.colsample_bytree is not None:
        custom_params['colsample_bytree'] = args.colsample_bytree
    if args.subsample is not None:
        custom_params['subsample'] = args.subsample
    
    try:
        print("🚀 Starting Hand Gesture Recognition Training")
        print("=" * 60)
        print("Following your exact notebook steps:")
        print("1. Load data from CSV")
        print("2. Split data (stratified)")
        print("3. Apply custom scaling (wrist-centered)")
        print("4. Encode labels")
        print("5. Train XGBoost with best parameters")
        print("6. Evaluate and save results")
        print("=" * 60)
        
        # Run experiment
        if trainer is not None:
            results = trainer.run_experiment(
                csv_path=args.csv,
                test_size=args.test_size,
                random_state=args.random_state,
                use_best_params=use_best_params,
                **custom_params
            )
        else:
            # Fallback without MLflow
            trainer = HandGestureTrainer()
            results = trainer.run_experiment_without_mlflow(
                csv_path=args.csv,
                test_size=args.test_size,
                random_state=args.random_state,
                use_best_params=use_best_params,
                **custom_params
            )
        
        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print("="*60)
        print("📁 Saved artifacts:")
        print(f"   Model: {results['model_path']}")
        print(f"   Label Encoder: {results['encoder_path']}")
        print(f"   Classes: {results['classes_path']}")
        print()
        print("🔄 Next steps:")
        print("1. 📊 View results in MLflow (if available):")
        print("   mlflow ui --backend-store-uri ./mlflow_data")
        print("2. 🔮 Test your model:")
        print(f"   python inference.py --model {results['model_path']} --encoder {results['encoder_path']}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup MLflow server
        if trainer and not args.keep_mlflow:
            trainer.cleanup()
        elif trainer and args.keep_mlflow:
            print(f"🌐 MLflow server still running at: {trainer.mlflow_uri}")
            print("💡 To stop it later, press Ctrl+C or kill the process")


if __name__ == "__main__":
    main()
