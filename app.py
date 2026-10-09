
from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)


model = joblib.load(os.path.join("models", "spam_model.pkl"))
vectorizer = joblib.load(os.path.join("models", "vectorizer.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    transformed = vectorizer.transform([message])

    prediction = model.predict(transformed)[0]  # returns 0 or 1
    if prediction == 1:
        result = "Spam"
        result_class = "spam"
    else:
        result = "Not Spam"
        result_class = "not-spam"


    #result = "Spam" if prediction == 1 else "Not Spam"

    #return render_template('index.html', prediction=result, message=message)

    return render_template("index.html",prediction=result,result_class=result_class, message=message)


if __name__ == '__main__':
    app.run(debug=True)
