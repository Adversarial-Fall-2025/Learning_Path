from svm_feature_selection import sample_folder,sample,label_folder,labels,X,y
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
svm = svm.SVC(C=25.0, random_state=42)
model = svm.fit(X_train, y_train)
preds = svm.predict(X_test)

print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))

