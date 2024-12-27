from flask import Flask, request, render_template
import pickle
import os

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        message = request.form['message']
        data = [message]  # Convert to list

        # Transform input using the vectorizer
        vect = vectorizer.transform(data)

        # Make prediction
        prediction = model.predict(vect)[0]

        # Display result
        result = 'Fraud' if prediction == 1 else 'Not Fraud'
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
