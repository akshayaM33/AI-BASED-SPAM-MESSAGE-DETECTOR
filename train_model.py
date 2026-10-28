

import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Open the ZIP file
with zipfile.ZipFile("dataset/sms+spam+collection.zip", "r") as file:

    # Read the dataset
    with file.open("SMSSpamCollection") as data:
        df = pd.read_csv(
            data,
            sep="\t",
            header=None,
            names=["label", "message"]
        )

# Display the first 5 messages
print(df.head())
# Count spam and legitimate messages
print("\nMessage counts:")
print(df["label"].value_counts())


# Separate messages and labels
X = df["message"]
y = df["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display the sizes
print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))

# Convert text messages into numbers
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create the AI model
model = MultinomialNB()

# Train the model
model.fit(X_train_vectorized, y_train)

# Predict the labels of test messages
y_pred = model.predict(X_test_vectorized)

# Check model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Display detailed performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "model/spam_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
