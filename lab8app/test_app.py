# test_app.py
import requests
import json
import sys

def test_api(comment_text):
    # Update the port to 8001
    url = 'http://127.0.0.1:8001/predict'
    data = {'reddit_comment': comment_text}
    
    print(f"Sending request to {url}")
    print(f"Request data: {data}")
    
    try:
        response = requests.post(url, json=data)
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(json.dumps(result, indent=4))
            
            # Extract and display the probabilities
            probs = result['Predictions'][0]
            print(f"\nProbability of not removing: {probs[0]:.4f}")
            print(f"Probability of removing: {probs[1]:.4f}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # Default test comment
    test_comment = "Testing a comment."
    
    # If command line argument is provided, use it instead
    if len(sys.argv) > 1:
        test_comment = sys.argv[1]
    
    print(f"Testing API with comment: '{test_comment}'")
    test_api(test_comment)