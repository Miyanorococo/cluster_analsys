import pandas as pd
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets
import seaborn as sns
import pandas as pd
import cufflinks as cf
import plotly
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from scipy import genfromtxt


# https://www.sejuku.net/blog/63329
# https://ohke.hateblo.jp/entry/2017/07/28/230000


x2 = np.array([[-34.6,-34.8,-41,-42.7,-12.7,-1.7,17.2,156.1],
               [-48.1,-51.5,-55.7,-59.4,-20.4,14.1,-0.8,84.6],
               [-46.1,-48.6,-54.2,-57,-11.5,7.7,14,233.1],
               [-57.6,-62,-57.4,-69.9,-28.9,20.1,40.1,81.5],
            [-60.6,-65.6,-69.2,-73.4,-25.9,10.4,31.2,121.7],
            [-51.7,-56,-51,-61.2,-17.3,46.2,30.7,2.4],
            [-54.6,-58.4,-58.2,-66.7,-28.3,18.8,52.2,98.7],
            [-48.3,-52,-54.9,-66.2,3.3,37.2,9.4,98.4],
                 [-47.5,-51.1,-52.4,-59.8,-21,22.6,55.7,108],
                 [-44.1,-51.1,-53.1,-62.1,-8.9,26.6,24.8,98.8],
                 [-12.2,-14.2,-16.5,-15.4,-55.1,-43.2,-62.2,20.3],
                 [-21.1,-22.9,-22.7,-30.5,-20.4,-13.3,1.7,163.7],
                 [-11.6,-13.6,-11.5,-21.4,-6.2,33,13,187],
                 [-14.8,-16.2,-10.2,-20.2,5,37,-2,41],
                 [-6.3,-8.8,-2.9,-16.7,-1,10,27,66],
                 [-39.4,-39.1,-38.8,-39.9,-33,-7,-5,103],
                 [-18,-22.8,-25.7,-29.5,3.7,6,57,77],
                 [-56.7,-53.5,-53.1,-53.7,31,-12,48,137],
                 [-22.5,-23.3,-25.6,-28.6,25,4,25,167],
                 [-35.3,-37.7,-40.3,-40.2,46,-15,36,125],
                 [-4.4, -9.9, -7.7, -8.2, -7.1, -6.3, 0, -3],
                 [-0.4, -6.5, 1.5, -5.81, 0.1, -1.2, 0, -0.6],
                 [-5, -8.3, 9, - 8.1, 2.9, -2.5, 0, -4.1],
                 [-3.8, -8.2, 10.6, -8.04, 1.7, -1.3, 0, -1.5],
                 [-9.2, -9.5, 19.3, -9.03, 3.6, 1.2, 0, -5.8],
                 [-6.3, -9.7, 9.4, -9.54, 4.7, -2.6, 0, -3.4],
                 [-3.6, -8.1, 5.4, -7.4, 7.6, -0.7, 0, -1.4],
                 [-7.9, -10, 8.4, -9.83, 9, -1.6, 0, -3.9],
                 [-3.1, -5.6, 16.6, -6.91, 11.9, -0.8, 0, -1.5],
                 [-6.8, -9, 8.7, -9.35, 10.5, -2.6, 0, -3.1],
                 [2.5, 1.9, -1.8, 2.5, -2.1, 1.4, 0, -3.7],
                 [-18.2, -17.9, -20, -24.2, -15.5, -2.6, 0, -11.8],
                 [-3.9, -9, -11.1, -5.9, 13.9, -5.1, 0, 0.6],
                 [-23.7, -18.5, -22.3, -18.7, -2.8, -12.3, 0, -16],
                 [-7.6, -6.5, -12.7, -10.2, 22.8, -5.5, 0, -8.8],
                 [-22, -20.8, -3.5, -21.3, -13.4, -11.8, 0, -15.5],
                 [4.6, 1.3, 47, -0.9, 21.9, 6.4, 0, -1.9],
                 [-13.1, -13.6, 22, -12.5, -9.5, -5.1, 0, -7.2],
                 [-4.6, -6.6, 48.9, -7.5, -3, 2.3, 0, -4.4],
                 [-8.3, -11.9, 45.2, -9.7, -1.9, -8.7, 0, -3.8],
                 [1.82, 1.92, 0.52, -2.97, -0.68, 1.12, 4.32, 0.94],
                 [0.53, 1.23, 0.53, -2.57, -0.27, 1.83, 3.53, 0.63],
                 [0.81, 0.81, 1.15, -2.35, -0.49, 1.81, 4.81, 0.51],
                 [1.36, 1.64, 0.74, -2.38, -0.56, 1.84, 3.84, 0.24],
                 [0.84, 0.44, 0.64, -2.47, -0.66, 1.74, 3.94, 0.34],
                 [1.1, 2.9, 0.6, -1.67, 0.6, 1.5, 3.4, 1.4],
                 [2.2, 2.6, 0.4, -1.48, 0.8, 2.5, 3.8, 1.4],
                 [0.8, 3.4, 0.2, -2.37, 1, 2, 2.8, 1.3],
                 [1.5, 3.9, -0.1, -2.06, 0.8, 1.7, 2.4, 1.7],
                 [1.5, 3.9, 0.3, -1.96, 1.4, 1.6, 2.7, 1.5],
                 [4.7, 1.1, 1.2, -0.48, -0.2, -0.9, -0.1, 1.9],
                 [4.7, -0.1, 1.8, -1.47, -1.1, 1.5, 0.4, 2.6],
                 [6.7, 1.6, 2.8, -0.49, -0.2, 3.5, 0.8, 4.4],
                 [1.9, 1.1, 0.4, -1.75, -0.2, 1.8, 1.4, 1.7],
                 [4.4, 1.8, 2.9, -0.2, 0.5, 4.6, 3.2, 0.4],
                 [-0.5, 2.5, -0.2, -2.3, -1.1, 1.9, 2.1, 2],
                 [0.8, 5.2, 1.4, 0.38, 1.5, 3.1, 1.1, 1.6],
                 [0.8, -0.9, 1.4, -1.5, -1, 2.7, 2.4, 2.6],
                 [2, 5.3, 2.9, 0.1, 1.5, 2.8, 1.9, 1.4],
             [-3.5, -1.8, 0.1, -2.6, -0.4, 1.1, 2, 0.8]])


y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print(x2.shape)
print(x2)
print(y.shape)
print('x2 shape: {}, y shape: {}'.format(x2.shape, y.shape))

# Split learn and test data
X_train, X_test, y_train, y_test = train_test_split(x2, y, random_state=0)

# generalize
std_scl = StandardScaler()
std_scl.fit(X_train)
X_train = std_scl.transform(X_train)
X_test = std_scl.transform(X_test)

# learning and test
svc = SVC()
svc.fit(X_train, y_train)

print('Train score: {:.3f}'.format(svc.score(X_train, y_train)))
print('Test score: {:.3f}'.format(svc.score(X_test, y_test)))

print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, svc.predict(X_test))))

# data set
# pandas.plotting.scatter_matrix(pd.DataFrame(x2), alpha=0.3, figsize=(16,16))
df = pd.DataFrame(x2)
#df["label"] = y
#df["label"] = ["compress","compress","compress","compress","compress","compress","compress","compress","compress","compress","twist","twist","twist","twist","twist","twist","twist","twist","twist","twist"]
df["label"] = ["compress","compress","compress","compress","compress","compress","compress","compress","compress","compress","twist","twist","twist","twist","twist","twist","twist","twist","twist","twist","compress","compress","compress","compress","compress","compress","compress","compress","compress","compress","twist","twist","twist","twist","twist","twist","twist","twist","twist","twist","compress","compress","compress","compress","compress","compress","compress","compress","compress","compress","twist","twist","twist","twist","twist","twist","twist","twist","twist","twist"]
print(df['label'].value_counts())
#sns.pairplot(df.iloc[:,:-1])
#sns.pairplot(df.iloc[10:20,:-1])
sns.pairplot(df, hue='label')
plt.show()

# PCA
pca = PCA(n_components=3)
X = pca.fit_transform(df.iloc[:,:-1].values)
embed3 = pd.DataFrame(X)
embed3["label"] = df["label"]
print(embed3)


# figure PCA
fig = pyplot.figure()
ax = Axes3D(fig)

#ax.plot(X[:10,0], X[:10,1], X[:10,2], "o", color="red", ms=4, mew=0.5)
#ax.plot(X[10:,0], X[10:,1], X[10:,2], "o", color="blue", ms=4, mew=0.5)
ax.plot(X[:10,0], X[:10,1], X[:10,2], "o",color="#004eff", ms=4, mew=0.5)
ax.plot(X[10:20,0], X[10:20,1], X[10:20,2], "o",color="#ff0000", ms=4, mew=0.5)
ax.plot(X[20:30,0], X[20:30,1], X[20:30,2], "o",color="#5a00ff", ms=4, mew=0.5)
ax.plot(X[30:40,0], X[30:40,1], X[30:40,2], "o",color="#ff0084", ms=4, mew=0.5)
ax.plot(X[40:50,0], X[40:50,1], X[40:50,2], "o",color="#00deff", ms=4, mew=0.5)
ax.plot(X[50:60,0], X[50:60,1], X[50:60,2], "o",color="#ff5a00", ms=4, mew=0.5)
pyplot.show()

X2 = PCA(n_components=2).fit_transform(df.iloc[:,:-1].values)

embed2 = pd.DataFrame(X2)
embed2["label"] = df["label"]
print(embed2)

#plt.plot(X2[:10,0], X2[:10,1], "o")
#plt.plot(X2[10:,0], X2[10:,1], "o")
plt.plot(X2[:10,0], X2[:10,1],  "o", color="#004eff")
plt.plot(X2[10:20,0], X2[10:20,1], "o", color="#ff0000")
plt.plot(X2[20:30,0], X2[20:30,1], "o", color="#5a00ff")
plt.plot(X2[30:40,0], X2[30:40,1], "o", color="#ff0084")
plt.plot(X2[40:50,0], X2[40:50,1], "o", color="#00deff")
plt.plot(X2[50:60,0], X2[50:60,1], "o", color="#ff5a00")
plt.show()

