from metaflow import FlowSpec, step, Parameter, Flow, conda_base, kubernetes, resources, timeout, retry, catch
import json
import numpy as np
import pickle
import os

@conda_base(libraries={'numpy': '1.23.5', 'scikit-learn': '1.2.2'}, python='3.9.16')
class ModelScoringFlowGCP(FlowSpec):
    # Define parameters for the flow
    input_data = Parameter('input_data',
                          help='Input data as a JSON string of feature values',
                          required=True)

    @step
    def start(self):
        """
        Start the flow by loading the input data and the latest training run
        """
        print("Starting the model scoring flow")

        # Get the latest training run to extract the scaler and feature information
        self.train_run = Flow('ModelTrainingFlowGCP').latest_run

        # Parse input data
        self.input_features = json.loads(self.input_data)

        # Validate that we have the correct number of features
        feature_names = self.train_run['start'].task.data.feature_names
        if len(self.input_features) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, but got {len(self.input_features)}")

        # Convert to numpy array
        self.features_array = np.array([self.input_features])

        print("Input data loaded successfully")

        # Move to preprocessing step
        self.next(self.preprocess_data)

    @step
    def preprocess_data(self):
        """
        Apply the same preprocessing as in the training flow
        """
        print("Preprocessing input data...")
        # Get the scaler from the training flow
        scaler = self.train_run['preprocess_data'].task.data.scaler

        # Scale the input features
        self.scaled_features = scaler.transform(self.features_array)

        print("Features preprocessed successfully")

        # Load the model and make predictions
        self.next(self.predict)

    @step
    @kubernetes
    @resources(cpu=1, memory="4GB")
    @timeout(minutes=10)
    @retry(times=2, delay=30)
    @catch(var='exception')
    def predict(self):
        """
        Load the model and make predictions
        """
        print("Loading model and making predictions...")
        try:
            # Get the model URI from the training run
            model_uri = self.train_run['register_model'].task.data.model_uri

            # Load model from local file
            with open(model_uri, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_uri}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load the model directly from the training run...")

            # Load model directly from the best_model attribute
            model = self.train_run['choose_best_model'].task.data.best_model
            print("Model loaded directly from training run")

        # Make predictions
        self.prediction = model.predict(self.scaled_features)[0]

        # Get the target names from the training run
        target_names = self.train_run['start'].task.data.target_names
        self.predicted_class = target_names[self.prediction]

        print(f"Prediction made: class {self.prediction} ({self.predicted_class})")

        if self.exception:
            print(f"Prediction failed: {self.exception}")
            self.next(self.end)  # Or a dedicated error handling step
        else:
            # Move to the end step
            self.next(self.end)

    @step
    def end(self):
        """
        End the flow and summarize the prediction
        """
        print("\n=== Model Scoring Flow Complete ===")
        print(f"Prediction: {self.prediction} ({self.predicted_class})")
        print(f"Input features: {self.input_features}")


if __name__ == '__main__':
    ModelScoringFlowGCP()