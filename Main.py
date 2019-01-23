from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
from Dictionary_n_Extraction import *
import numpy as np

#Preparing training data
train_labels = np.zeros(5175)
train_labels[1501:5175] = 1
train_matrix = extract(train_dir)

#Training  classifier
model = MultinomialNB()
model.fit(train_matrix,train_labels)
model2 = LogisticRegression()
model2.fit(train_matrix,train_labels)

# Test
test_dir = 'test-mails'
test_matrix = extract(test_dir)
test_labels = np.zeros(6000)
test_labels[1501:6000] = 1
result = model.predict(test_matrix)
result2 = model2.predict(test_matrix)

print("For Bayes Classifier: ")
print("Confusion Matrix: ")
print(confusion_matrix(test_labels,result))
print("\nAccuracy score: ")
print(accuracy_score(test_labels,result))

print("\nFor Logistic Regression Classifier: ")
print("Confusion Matrix: ")
print(confusion_matrix(test_labels,result2))
print("\nAccuracy score: ")
print(accuracy_score(test_labels,result2))