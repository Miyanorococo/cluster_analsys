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

x2 = np.array([[-4.4,-9.9,-7.7,-8.2,-7.1,-6.3,0.01,-3],
             [-0.4,-6.5,1.5,-5.81,0.1,-1.2,0.01,-0.6],
             [-5,-8.3,9,-8.1,2.9,-2.5,0.01,-4.1],
             [-3.8,-8.2,10.6,-8.04,1.7,-1.3,0.01,-1.5],
             [-9.2,-9.5,19.3,-9.03,3.6,1.2,0.01,-5.8],
             [-6.3,-9.7,9.4,-9.54,4.7,-2.6,0.01,-3.4],
             [-3.6,-8.1,5.4,-7.4,7.6,-0.7,0.01,-1.4],
             [-7.9,-10,8.4,-9.83,9,-1.6,0.01,-3.9],
             [-3.1,-5.6,16.6,-6.91,11.9,-0.8,0.01,-1.5],
             [-6.8,-9,8.7,-9.35,10.5,-2.6,0.01,-3.1],
             [2.5,1.9,-1.8,2.5,-2.1,1.4,0.01,-3.7],
             [-18.2,-17.9,-20,-24.2,-15.5,-2.6,0.01,-11.8],
             [-3.9,-9,-11.1,-5.9,13.9,-5.1,0.01,0.6],
             [-23.7,-18.5,-22.3,-18.7,-2.8,-12.3,0.01,-16],
             [-7.6,-6.5,-12.7,-10.2,22.8,-5.5,0.01,-8.8],
             [-22,-20.8,-3.5,-21.3,-13.4,-11.8,0.01,-15.5],
             [4.6,1.3,47,-0.9,21.9,6.4,0.01,-1.9],
             [-13.1,-13.6,22,-12.5,-9.5,-5.1,0.01,-7.2],
             [-4.6,-6.6,48.9,-7.5,-3,2.3,0.01,-4.4],
             [-8.3,-11.9,45.2,-9.7,-1.9,-8.7,0.01,-3.8]])

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

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
df["label"] = ["compress","compress","compress","compress","compress","compress","compress","compress","compress","compress","twist","twist","twist","twist","twist","twist","twist","twist","twist","twist"]
print(df['label'].value_counts())
#sns.pairplot(df.iloc[:,:-1])
#sns.pairplot(df.iloc[10:20,:-1])
sns.pairplot(df, hue="label")
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

ax.plot(X[:10,0], X[:10,1], X[:10,2], "o", color="red", ms=4, mew=0.5)
ax.plot(X[10:,0], X[10:,1], X[10:,2], "o", color="blue", ms=4, mew=0.5)
pyplot.show()

X2 = PCA(n_components=2).fit_transform(df.iloc[:,:-1].values)

embed2 = pd.DataFrame(X2)
embed2["label"] = df["label"]
print(embed2)

plt.plot(X2[:10,0], X2[:10,1], "o", color="red")
plt.plot(X2[10:,0], X2[10:,1], "o", color="blue")
plt.show()

