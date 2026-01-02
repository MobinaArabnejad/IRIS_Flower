import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import ( accuracy_score, confusion_matrix, classification_report)
#writing x
iris_bunch = load_iris()
X = pd.DataFrame(iris_bunch.data, columns=iris_bunch.feature_names)
y = pd.Series(iris_bunch.target, name="species")
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
print("cm :\n", cm)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap='Greens', fmt='d',
            xticklabels=iris_bunch.target_names,
            yticklabels=iris_bunch.target_names)
plt.xlabel('predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.tight_layout( )
plt.show()
