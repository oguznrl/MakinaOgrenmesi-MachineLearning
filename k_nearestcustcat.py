import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df=pd.read_csv('teleCust1000t.csv')
df.head()

print(df['custcat'].value_counts())

#Görselleştirme tekniklerini kullanarak verilerinizi kolayca keşfedebilirsiniz

df.hist(column='income',bins=50)

X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
y = df[['custcat']]

#data normalizasyon

X=preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


#sınıflandırma

from sklearn.neighbors import KNeighborsClassifier

k=4
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)

yhat=neigh.predict(X_test)
print("x test ",X_test,"yhat ",yhat)
from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))