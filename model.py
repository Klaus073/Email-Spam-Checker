import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix 
from sklearn.feature_extraction.text import CountVectorizer

# Load the CSV file
data = pd.read_csv('encoded_emails.csv')

# Split the data into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data['Message'],  # Features (emails)
    data['Category'],  # Labels
    test_size=0.2,  # 80% for training, 20% for testing
    random_state=42
)

count_vectorizer = CountVectorizer()
X_train_counts = count_vectorizer.fit_transform(train_data)
X_test_counts = count_vectorizer.transform(test_data)

# Train the Logistic Regression model
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_counts, train_labels)

# Evaluate the model on the test set
accuracy = logreg_model.score(X_test_counts, test_labels)
print("Logistic Regression Accuracy:", accuracy)

# Perform k-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracy_scores = []


for train_index, val_index in kf.split(X_train_counts):
    # Split the data into train and validation sets for this fold
    fold_train_data, fold_val_data = X_train_counts[train_index], X_train_counts[val_index]
    fold_train_labels, fold_val_labels = train_labels.iloc[train_index], train_labels.iloc[val_index]

    # Fit the model on the fold training data
    logreg_model.fit(fold_train_data, fold_train_labels)

    # Make predictions on the fold validation data
    fold_predictions = logreg_model.predict(fold_val_data)

    # Calculate accuracy for the fold
    fold_accuracy = accuracy_score(fold_val_labels, fold_predictions)

    # Append accuracy to list
    accuracy_scores.append(fold_accuracy)

# Calculate average accuracy across all folds
average_accuracy = sum(accuracy_scores) / k
print("Average Accuracy (k-fold cross-validation):", average_accuracy)



# Make predictions on the test set
test_predictions = logreg_model.predict(X_test_counts)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, test_predictions)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Function to preprocess the email
import re

def preprocess_email(email):
    # Remove non-alphanumeric characters
    email = re.sub(r'\W', ' ', email)

    # Convert to lowercase
    email = email.lower()

    return email


# Function to predict if an email is spam or not
def predict_spam(email):
    # Preprocess the email
    preprocessed_email = preprocess_email(email)

    # Transform the preprocessed email using the same CountVectorizer
    email_counts = count_vectorizer.transform([preprocessed_email])

    # Make a prediction
    prediction = logreg_model.predict(email_counts)

    return prediction[0]


# Example usage
# email = ""
# prediction = predict_spam(email)

# if prediction ==1:
#     print("This email is spam.")
# else:
#     print("This email is not spam.")

# List of sample emails
sample_emails = [
    "Get a discount on our exclusive products! Limited time offer!",
    "Dear customer, your account balance is low. Please top up your account.",
    "Hey, let's meet up for lunch tomorrow. What time works for you?",
    "Click this link to win a free vacation!",
    "Important notice: Your package has been delivered.",
    "Hi, I have an exciting business opportunity for you. Let's talk!",
    "You have won a lottery! Claim your prize now!",
    "Reminder: Your appointment is scheduled for tomorrow.",
    "Congratulations! You have been selected for a job interview.",
    "Earn $1000 per day with our proven money-making system!",
    "Dear valued customer, we appreciate your loyalty. Enjoy a special discount on your next purchase.",
    "URGENT: Your account security has been compromised. Please reset your password immediately.",
    "Invitation: Join us for an exclusive product launch event on Friday.",
    "Limited stock available! Buy now to get the best deal.",
    "Congratulations! You've won a free gift voucher. Claim it now.",
    "Reminder: Your subscription will expire soon. Renew now to continue enjoying our services.",
    "Important update: Changes to our terms and conditions. Please review and acknowledge.",
    "We are hiring! Apply now to join our dynamic team.",
    "Exciting offer: Get a free trial of our premium membership for 30 days.",
    "Emergency alert: Severe weather conditions in your area. Stay safe and follow instructions.",
    # Add more sample emails here...
]

# Initialize counters
not_spam_count = 0
not_spam_emails = []

spam_count = 0
spam_emails = []

# Predict if the emails are spam or not
for email in sample_emails:
    prediction = predict_spam(email)
    if prediction == 0:
        not_spam_count += 1
        not_spam_emails.append(email)
    else:
        spam_count += 1
        spam_emails.append(email)

# Print the total number of not spam emails and the list of those emails
print("Total Not Spam Emails:", not_spam_count)
print("Not Spam Emails:")
for email in not_spam_emails:
    print("-", email)

# Print the total number of  spam emails and the list of those emails
print("Total  Spam Emails:", spam_count)
print(" Spam Emails:")
for email in spam_emails:
    print("-", email)
