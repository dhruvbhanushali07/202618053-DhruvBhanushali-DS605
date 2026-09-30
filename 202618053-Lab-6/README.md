## Part A: Observation

The image features were extracted using grayscale intensity statistics and Canny edge information. Logistic Regression initially achieved 62.50% accuracy. After applying StandardScaler, its accuracy increased to 92.50%, showing that feature scaling was important for this model.

Random Forest achieved 93.75% accuracy, with a precision of 94.87%, recall of 92.50%, and F1-score of 93.67%. Its confusion matrix shows that it correctly classified 75 out of 80 test images.

Overall, scaling substantially improved Logistic Regression, while Random Forest provided strong performance using the extracted image features.

## Part B: Observation

The provided email dataset already contains 3000 word-frequency features for each email, so it was used directly as a bag-of-words representation rather than applying CountVectorizer again.

Multinomial Naive Bayes achieved 94.20% accuracy, while Logistic Regression achieved 98.26% accuracy on the same test set. Logistic Regression also achieved higher precision, recall, and F1-score, although it required more training time.

The results show that Logistic Regression performed better on the given word-frequency representation, while Multinomial Naive Bayes required less training time.

## Part C: Observation

To improve the image representation, 16 normalized grayscale histogram features were added to the original six image features. This increased the numerical feature set from 6 to 22 features.

The improved Random Forest achieved 96.25% accuracy and an F1-score of 96.30%, compared with 93.75% accuracy and 93.67% F1-score for the original Random Forest.

The improved model correctly classified 77 out of 80 test images, with only 3 misclassifications. This indicates that the additional histogram features provided useful information for distinguishing crack and non-crack images.