from svm_feature_selection import X,y
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
svm = svm.SVC(C=1.0, random_state=42)
model = svm.fit(X_train, y_train)

y_pred = model.predict(X)
print(classification_report(y, y_pred))
print(confusion_matrix(y, y_pred))
