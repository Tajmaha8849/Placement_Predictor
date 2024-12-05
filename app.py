import numpy as np
import pickle
from flask import Flask, request, render_template

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model1.pkl', 'rb'))

@app.route('/')
def home():
    # Render the home page
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering predictions on HTML form inputs.
    """
    # Retrieve input values from form
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Transform inputs for model prediction
    features = np.array([[cgpa, iq]])
    prediction = model.predict(features)[0]

    # Map prediction to output
    output = "Placed" if prediction == 1 else "Not Placed"
    improvement_message = "Improvement Required" if prediction == 0 else ""

    # Flag for modal
    show_modal = (prediction == 1)

    return render_template('index.html', prediction_text=f'Prediction: {output}', 
                           improvement_message=improvement_message, show_modal=show_modal)

if __name__ == '__main__':
    app.run(debug=True)
