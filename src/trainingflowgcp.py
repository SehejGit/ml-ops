# src/trainingflowgcp.py
from metaflow import FlowSpec, step, Parameter, conda_base, kubernetes, resources, timeout, retry, catch, current
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Don't import mlflow at the top level
# We'll import it conditionally when needed

@conda_base(libraries={'pandas': '1.5.3', 'numpy': '1.24.2', 'scikit-learn': '1.2.2'}, python='3.9.16')
class ModelTrainingFlowGCP(FlowSpec):
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
                                  
    use_mlflow = Parameter('use_mlflow',
                         help='Whether to use MLflow for model tracking',
                         default=False,
                         type=bool)

    @step
    def start(self):
        """
        Start the flow by loading and preprocessing the data
        """
        print("Starting the model training flow")
        
        # Check if we're running in Kubernetes
        is_kubernetes = 'kubernetes' in current.environment
        
        # Create synthetic car price data
        np.random.seed(self.random_state)
        n_samples = 500
        
        # Generate synthetic data
        years = np.random.randint(2000, 2023, n_samples)
        engine_sizes = np.random.uniform(1.0, 6.0, n_samples)
        horsepowers = np.random.randint(100, 500, n_samples)
        torques = horsepowers * 0.8 + np.random.normal(0, 20, n_samples)
        acceleration = 10 - (horsepowers / 100) + np.random.normal(0, 1, n_samples)
        
        # Generate price as a function of the other features + some noise
        prices = (2023 - years) * -500 + engine_sizes * 2000 + horsepowers * 50 + torques * 10 - acceleration * 2000
        prices = prices + np.random.normal(0, 5000, n_samples)
        prices = np.maximum(prices, 5000)  # Ensure no negative prices
        
        # Create DataFrame
        df = pd.DataFrame({
            'Year': years,
            'Engine Size (L)': engine_sizes,
            'Horsepower': horsepowers,
            'Torque (lb-ft)': torques,
            '0-60 MPH Time (seconds)': acceleration,
            'Price (in USD)': prices
        })
        
        print("Synthetic dataset created successfully")
        
        # Display info about the dataset
        print(f"Dataset shape: {df.shape}")
        
        # Select features and target
        X = df[['Year', 'Engine Size (L)', 'Horsepower', 'Torque (lb-ft)', '0-60 MPH Time (seconds)']]
        y = df['Price (in USD)']
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        self.target_name = 'Price (in USD)'

        # Split the data into training and test sets
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        print(f"Data loaded and split: {self.train_data.shape[0]} training samples, {self.test_data.shape[0]} test samples")

        # Next, let's scale the features
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Preprocess the data by scaling features
        """
        # Scale the features
        scaler = StandardScaler()
        self.train_data_scaled = scaler.fit_transform(self.train_data)
        self.test_data_scaled = scaler.transform(self.test_data)

        # Store the scaler for later use
        self.scaler = scaler

        print("Features scaled successfully")

        # Train different models in parallel
        self.next(self.train_random_forest, self.train_svm)

    @catch(var='exception')
    @retry(times=3, minutes_between_retries=1)
    @timeout(minutes=30)
    @kubernetes
    @resources(cpu=2, memory=8192)  # 8GB = 8192MB
    @step
    def train_random_forest(self):
        """
        Train a Random Forest regressor
        """
        print("Training Random Forest...")
        
        if not hasattr(self, 'exception') or not self.exception:
            # Train a Random Forest regressor
            rf = RandomForestRegressor(random_state=self.random_state)
            rf.fit(self.train_data_scaled, self.train_labels)

            # Evaluate the model
            y_pred = rf.predict(self.test_data_scaled)
            mse = mean_squared_error(self.test_labels, y_pred)
            r2 = r2_score(self.test_labels, y_pred)

            # Store the model and metrics
            self.model = rf
            self.model_type = 'RandomForest'
            self.metrics = {
                'mse': mse,
                'r2_score': r2
            }

            print(f"Random Forest trained with MSE: {mse:.4f}, R² score: {r2:.4f}")
        else:
            print(f"Random Forest training failed: {self.exception}")
            
        # Always include an unconditional self.next() at the end of the step
        self.next(self.choose_best_model)

    @catch(var='exception')
    @retry(times=3, minutes_between_retries=1)
    @timeout(minutes=30)
    @kubernetes
    @resources(cpu=2, memory=8192)  # 8GB = 8192MB
    @step
    def train_svm(self):
        """
        Train a Support Vector Machine regressor
        """
        print("Training SVM...")
        
        if not hasattr(self, 'exception') or not self.exception:
            # Train an SVM regressor
            svm = SVR()
            svm.fit(self.train_data_scaled, self.train_labels)

            # Evaluate the model
            y_pred = svm.predict(self.test_data_scaled)
            mse = mean_squared_error(self.test_labels, y_pred)
            r2 = r2_score(self.test_labels, y_pred)

            # Store the model and metrics
            self.model = svm
            self.model_type = 'SVM'
            self.metrics = {
                'mse': mse,
                'r2_score': r2
            }

            print(f"SVM trained with MSE: {mse:.4f}, R² score: {r2:.4f}")
        else:
            print(f"SVM training failed: {self.exception}")
        
        # Always include an unconditional self.next() at the end of the step
        self.next(self.choose_best_model)

    @step
    def choose_best_model(self, inputs):
        """
        Select the best model based on R² score
        """
        print("Choosing the best model...")
        # Choose the model with the highest R² score
        self.models = {inp.model_type: (inp.model, inp.metrics) for inp in inputs if hasattr(inp, 'model_type')}
        
        if not self.models:
            print("No models were successfully trained. Exiting.")
            self.best_model = None
            self.best_model_name = None
            self.best_metrics = {'mse': float('inf'), 'r2_score': -float('inf')}
        else:
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k][1]['r2_score'])

            # Get the best model and its metrics
            self.best_model, self.best_metrics = self.models[best_model_name]
            self.best_model_name = best_model_name

            print(f"Best model: {best_model_name} with R² score: {self.best_metrics['r2_score']:.4f}")

            # Explicitly set the conflicting attributes
            self.model = self.best_model
            self.model_type = self.best_model_name

        # Now merge the remaining artifacts that don't conflict
        self.merge_artifacts(inputs)

        # Register the model with MLFlow or save locally
        self.next(self.register_model)

    @step
    def register_model(self):
        """
        Register the best model or save locally
        """
        print("Registering the model...")
        
        if self.best_model is None:
            print("No model to register. Exiting.")
            self.model_uri = None
        else:
            # Only try MLflow if explicitly requested and not in Kubernetes
            is_kubernetes = 'kubernetes' in current.environment
            use_mlflow_here = self.use_mlflow and not is_kubernetes
            
            if use_mlflow_here:
                try:
                    # Import MLflow only when needed
                    import mlflow
                    
                    # Set up MLFlow tracking
                    print(f"Attempting to connect to MLFlow server at {self.mlflow_tracking_uri}")
                    mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                    mlflow.set_experiment("car-price-prediction")

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
                            registered_model_name="car-price-predictor"
                        )

                        # Store the model URI for later use
                        self.model_uri = model_info.model_uri

                        print(f"Model registered with MLFlow: {self.model_uri}")

                except Exception as e:
                    print(f"Error with MLflow: {e}")
                    print("Saving model locally instead...")
                    self._save_model_locally()
            else:
                # Save locally
                print("MLflow not enabled or running in Kubernetes. Saving model locally.")
                self._save_model_locally()

        # Move to the end step
        self.next(self.end)
        
    def _save_model_locally(self):
        """Helper method to save the model locally"""
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

    @step
    def end(self):
        """
        End the flow and summarize results
        """
        print("\n=== Model Training Flow Complete ===")
        if self.best_model is None:
            print("No model was successfully trained.")
        else:
            print(f"Best model: {self.best_model_name}")
            print(f"Model metrics: {self.best_metrics}")
            print(f"Model URI: {self.model_uri}")
            print("You can now use the scoring flow to make predictions with this model.")


if __name__ == '__main__':
    ModelTrainingFlowGCP()