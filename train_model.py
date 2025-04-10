# 1. Import Libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 2. Load Cleaned Data
df = pd.read_csv('Scripts/CleanedMisogynisticTweets.csv')

# 3. TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['text'])
y = df['label']

print("✅ TF-IDF vectorization complete!")

# 4. Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Train/test split complete!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 5. Train Logistic Regression Model
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

# 6. Evaluate Logistic Regression
y_pred_lr = lr_model.predict(X_test)

print("\n✅ Logistic Regression Model Trained!")
print("\nLogistic Regression - Accuracy Score:", accuracy_score(y_test, y_pred_lr))
print("\nLogistic Regression - Classification Report:\n", classification_report(y_test, y_pred_lr))
print("\nLogistic Regression - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# 7. Train Support Vector Machine (SVM) Model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# 8. Evaluate SVM
y_pred_svm = svm_model.predict(X_test)

print("\n✅ SVM Model Trained!")
print("\nSVM - Accuracy Score:", accuracy_score(y_test, y_pred_svm))
print("\nSVM - Classification Report:\n", classification_report(y_test, y_pred_svm))
print("\nSVM - Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))

# 9. Save Models and Vectorizer
joblib.dump(lr_model, 'harassment_detector_model.pkl')
joblib.dump(svm_model, 'svm_detector_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\n✅ Models and TF-IDF vectorizer saved successfully!")