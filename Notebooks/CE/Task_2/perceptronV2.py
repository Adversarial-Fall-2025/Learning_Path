import numpy as np
from svm_feature_selection import X, y
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
	def __init__(self, features, learning_rate = 0.1, epochs = 50):
		
		self.features = features
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = np.zeros(features)
		self.bias = 0.01
	
	def predict(self, x):
		x = np.array(x)
		preds = []

		if x.ndim == 1:
			prediction = np.dot(x, self.weights) + self.bias
			return 1 if prediction > 0 else 0
		else:
			for xi in x:

				prediction = np.dot(xi, self.weights) + self.bias
				preds.append(1 if prediction > 0 else 0)
			return np.array(preds)

	def fit(self, X, y):
		
		for i in range(self.epochs):
			for xi, yval in zip(X, y):
				prediction = self.predict(xi)
				error = yval - prediction

				self.weights += self.learning_rate * error * xi
				self.bias += self.learning_rate * error
	

X, y = X, y
p = Perceptron(2)
p.fit(X, y)
ypred = p.predict(X)
#X_train, y_train, X_test, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)
	#Train-test split under development
#p.fit(X_train,y_train)
#print(p.predict(X[0]))
#ypred = p.predict(X_test)
#print(f"y shape: {y.shape}\nX shape: {X.shape}\nypred shape: {ypred.shape}")
print(confusion_matrix(y, ypred))
print(classification_report(y, ypred))
'''
vals = pd.DataFrame({'NDVI-Mean': X[:, 0],
			'NDVI-STD': X[:, 1],
			'True': y,
			'Predicted':ypred
		})

sns.scatterplot(data=vals,
		x='NDVI-Mean',
		y='NDVI-STD',
		hue='Predicted',
		style='True',
		palette={0: 'red', 1: 'green'},
		alpha=0.7
	)

plt.title('Preds vs True Vals')
plt.xlabel('NDVI-Mean')
plt.ylabel('NDVI-STD')

w = p.weights
b = p.bias
x0 = -b/w[0]
plt.axvline(x=x0, color='blue', linestyle='--', label=f'Boundary (x={x0:.3f})')
plt.legend()
plt.savefig('ndvi_perceptron_boundary.png', dpi=300, bbox_inches='tight')
plt.close()

#print(f"weights: {w}\n bias: {b}\nPrediction: {ypred}")
'''
