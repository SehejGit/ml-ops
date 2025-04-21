# src/trainingflow.py
from metaflow import FlowSpec, step, Parameter
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Import our helper function
from data_prep import get_wine_data

class ModelTrainingFlow(FlowSpec):
    # Define parameters for the flow
    random_state = Parameter('random_state', 
                            help='Random seed for reproducibility',
                            default=42, 
                            type=int)
    
    test_size = Parameter('test_size',
                        help='Proportion of data to use for testing',
                        default=0.2,
                        type=float)
    
    # MLFlow tracking URI - replace with your own when needed
    mlflow_tracking_uri = Parameter('mlflow_uri',
                                  help='MLFlow tracking URI',
                                  default='http://localhost:5000',
                                  type=str)

    @step
    def start(self):
        """
        Start the flow by loading and preprocessing the data
        """
        print("Starting the model training flow")
        
        # Load the data using our helper function
        X, y, feature_names, target_names = get_wine_data()
        
        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Store feature names and target names for later use
        self.feature_names = feature_names
        self.target_names = target_names
        
        print(f"Data loaded and split: {self.X_train.shape[0]} training samples, {self.X_test.shape[0]} test samples")
        
        # Next, let's scale the features
        self.next(self.preprocess_data)
        
    @step
    def preprocess_data(self):
        """
        Preprocess the data by scaling features
        """
        # Scale the features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Store the scaler for later use
        self.scaler = scaler
        
        print("Features scaled successfully")
        
        # Train different models in parallel
        self.next(self.train_random_forest, self.train_svm)
        
    @step
    def train_random_forest(self):
        """
        Train a Random Forest classifier
        """
        # Train a Random Forest classifier
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate the model
        y_pred = rf.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Store the model and metrics
        self.model = rf
        self.model_type = 'RandomForest'
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"Random Forest trained with accuracy: {accuracy:.4f}, f1 score: {f1:.4f}")
        
        # Move to the next step
        self.next(self.choose_best_model)
        
    @step
    def train_svm(self):
        """
        Train a Support Vector Machine classifier
        """
        # Train an SVM classifier
        svm = SVC(random_state=self.random_state)
        svm.fit(self.X_train_scaled, self.y_train)
        
        # Evaluate the model
        y_pred = svm.predict(self.X_test_scaled)
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        # Store the model and metrics
        self.model = svm
        self.model_type = 'SVM'
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        print(f"SVM trained with accuracy: {accuracy:.4f}, f1 score: {f1:.4f}")
        
        # Move to the next step
        self.next(self.choose_best_model)
        
    @step
    def choose_best_model(self, inputs):
        """
        Select the best model based on F1 score
        """
        # Choose the model with the highest F1 score
        self.models = {inp.model_type: (inp.model, inp.metrics) for inp in inputs}
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k][1]['f1_score'])
        
        # Get the best model and its metrics
        self.best_model, self.best_metrics = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"Best model: {best_model_name} with F1 score: {self.best_metrics['f1_score']:.4f}")
        
        # Explicitly set the conflicting attributes
        self.model = self.best_model
        self.model_type = self.best_model_name
        
        # Now merge the remaining artifacts that don't conflict
        self.merge_artifacts(inputs)
        
        # Register the model with MLFlow
        self.next(self.register_model)
        
    @step
    def register_model(self):
        """
        Register the best model with MLFlow or save locally if MLFlow server isn't available
        """
        try:
            # Set up MLFlow tracking
            print(f"Attempting to connect to MLFlow server at {self.mlflow_tracking_uri}")
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment("wine-classification")
            
            # Log the model and metrics
            with mlflow.start_run(run_name=f"{self.best_model_name}_training"):
                # Log parameters
                mlflow.log_param("model_type", self.best_model_name)
                mlflow.log_param("random_state", self.random_state)
                mlflow.log_param("test_size", self.test_size)
                
                # Log metrics
                for metric_name, metric_value in self.best_metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log the model
                model_info = mlflow.sklearn.log_model(
                    self.best_model, 
                    "model",
                    registered_model_name="wine-classifier"
                )
                
                # Store the model URI for later use
                self.model_uri = model_info.model_uri
                
                print(f"Model registered with MLFlow: {self.model_uri}")
        
        except Exception as e:
            print(f"Error connecting to MLFlow: {e}")
            print("Saving model locally instead...")
            
            # Import pickle for local saving
            import pickle
            import os
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save the model locally
            model_path = f"models/{self.best_model_name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Save model metadata
            metadata_path = f"models/{self.best_model_name}_metadata.pkl"
            metadata = {
                'model_type': self.best_model_name,
                'random_state': self.random_state,
                'test_size': self.test_size,
                'metrics': self.best_metrics
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Set model URI to local path
            self.model_uri = os.path.abspath(model_path)
            print(f"Model saved locally: {self.model_uri}")
        
        # Move to the end step
        self.next(self.end)
        
    @step
    def end(self):
        """
        End the flow and summarize results
        """
        print("\n=== Model Training Flow Complete ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Model metrics: {self.best_metrics}")
        print(f"Model URI: {self.model_uri}")
        print("You can now use the scoring flow to make predictions with this model.")
        
if __name__ == '__main__':
    ModelTrainingFlow()