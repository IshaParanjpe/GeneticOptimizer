from genetic_library import LogisticRegression
import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=2345,
                           n_clusters_per_class=1)


clf = LogisticRegression()
plotter = clf.fit(X,y)


print(y)
print(clf.predict(X))



accuracy = np.sum(np.equal(y,clf.predict(X)))/len(y)
print(accuracy)


