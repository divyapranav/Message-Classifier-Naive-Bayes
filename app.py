from flask import Flask, render_template, request
import pickle

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)[0]
        
        result = "Spam" if prediction == 1 else "Ham (Not Spam)"
        return render_template('index.html', prediction_text=f"The message is: {result}")

if __name__ == "__main__":
    app.run(debug=True)
