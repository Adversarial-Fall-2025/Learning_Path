from svm_feature_selection import X,y
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
svm = svm.SVC(C=1.0, random_state=42)
model = svm.fit(X_train, y_train)
preds = svm.predict(X_test)

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(f"Training Accuracy: {model.score(X_train, y_train)}")
scores = cross_val_score(model, X, y, cv=5)
print("Cross-Validation scores: ", scores)
print(f"Average Accuracy: {scores.mean()}")
