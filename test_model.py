import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load('harassment_detector_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to predict a tweet
def predict_tweet(tweet_text):
    vectorized_text = vectorizer.transform([tweet_text])
    prediction = model.predict(vectorized_text)
    if prediction[0] == 1:
        print("This tweet is predicted as: Misogynistic (Label 1)\n")
    else:
        print("This tweet is predicted as: Normal (Label 0)\n")

# Keep checking tweets until the user decides to stop
print("Harassment Detector — Type a tweet to check, or type 'exit' to quit.")

while True:
    new_tweet = input("Enter a tweet for analysis: ")
    if new_tweet.lower() == 'exit':
        print("Exiting Harassment Detector. Goodbye!")
        break
    predict_tweet(new_tweet)
