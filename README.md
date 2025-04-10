AI-Based Harassment Detector for Social Media

Project Overview
This project presents a classical machine learning approach to detecting misogynistic and abusive content in social media posts. The solution focuses on lightweight, interpretable models rather than complex deep learning architectures, aiming for a balance between high accuracy and practical deployment.

The primary objective is to build a harassment detection system that identifies both blatant and subtle abusive content while maintaining transparency and computational efficiency.

Dataset Description
- Tweets were manually scraped and supplemented with synthetic examples to simulate realistic social media scenarios.
- The dataset was manually labeled into two classes:
  - 0: Normal (non-misogynistic) tweets
  - 1: Misogynistic or abusive tweets
- Preprocessing steps included:
  - Lowercasing all text
  - Removing punctuation, special characters, and excess whitespace
  - Standardizing text for consistency

Methodology
- Feature Extraction: Term Frequency-Inverse Document Frequency (TF-IDF) vectorization was applied to transform text data into numerical feature vectors.
- Models Trained:
  - Logistic Regression: Selected as the final model due to superior performance and interpretability.
  - Support Vector Machine (SVM): Trained for comparative analysis.
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix analysis

Results
Model: Logistic Regression
Accuracy: 97.2%

Model: Support Vector Machine (SVM)
Accuracy: 96.7%

Logistic Regression demonstrated slightly better overall performance and was selected for deployment.

How to Run the Project
Training (Optional)
- Execute train_model.py to retrain the models and re-save updated versions.
- This step is optional as pre-trained models are already included.

Testing
- Execute test_model.py
- Enter any tweet text when prompted.
- The model will predict:
  - 0: Normal content
  - 1: Misogynistic or abusive content
- Type exit to terminate the session.

Files Included
train_model.py - Script for model training and evaluation
test_model.py - Script for testing individual tweet inputs
harassment_detector_model.pkl - Saved Logistic Regression model
svm_detector_model.pkl - Saved Support Vector Machine model
tfidf_vectorizer.pkl - Saved TF-IDF vectorizer
CleanedMisogynisticTweets.csv - Final cleaned dataset
README.txt - Project overview and instructions

Future Improvements
- Incorporate sarcasm detection techniques to improve subtle harassment identification.
- Expand dataset across diverse social media platforms to enhance generalizability.
- Explore lightweight ensemble methods to boost performance further while maintaining model interpretability.

Project Contributor
Muslimah
University Project | 2025
#   M L 
 
 