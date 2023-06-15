from flask import Flask, render_template, request
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd


app = Flask(__name__)

# Load the trained model
logreg_model = LogisticRegression(max_iter=1000)
count_vectorizer = CountVectorizer()
data = pd.read_csv('encoded_emails.csv')
X_train_counts = count_vectorizer.fit_transform(data['Message'])
logreg_model.fit(X_train_counts, data['Category'])

# Preprocess email function
def preprocess_email(email):
    email = re.sub(r'\W', ' ', email.lower())
    return email

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    email = request.form['email']

    # Preprocess the email
    preprocessed_email = preprocess_email(email)

    # Transform the preprocessed email using the same CountVectorizer
    email_counts = count_vectorizer.transform([preprocessed_email])

    # Make a prediction
    prediction = logreg_model.predict(email_counts)

    if prediction == 1:
        result = "This email is spam."
    else:
        result = "This email is not spam."

    return render_template('index.html', email=email, result=result)



if __name__ == '__main__':
    app.run(debug=True)
