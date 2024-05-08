from flask import Flask, request
import your_prediction_function  # Import both functions

app = Flask(__name__)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes_api():
  # Get data from the request (assuming JSON format)
  data = request.get_json()
  
  # Check if data is present
  if not data:
    return "Error: No data provided", 400  # Bad request

  # Call the prediction function with the data
  prediction = your_prediction_function.predict_diabetes(data)
  
  # Get the message based on prediction
  message = your_prediction_function.get_diabetes_message(prediction)
  
  # Return JSON response with message
  response = {"message": message}
  
  return response

if __name__ == '__main__':
  app.run(debug=True)
