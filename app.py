from flask import Flask, render_template, request
import numpy as np
import pickle

# Load ML model
model = pickle.load(open('models/model.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

# Home Page
@app.route('/')
def home():
    return render_template('home.html')

# Predict Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    message = []
    if request.method == 'POST':
        features = request.form['feature']
        try:
            # Expecting comma-separated values
            features = features.split(',')
            np_features = np.asarray(features, dtype=np.float32)

            # Predict
            pred = model.predict(np_features.reshape(1, -1))
            message.append('Cancerous' if pred[0] == 1 else 'Not Cancerous')
        except:
            message.append("Invalid input. Please enter comma-separated numbers.")
    return render_template('predict.html', message=message)

# About Page
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))  # Railway provides PORT
    app.run(debug=False, host="0.0.0.0", port=port)
