from genetic_library import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1, bias=1, random_state=1000)


clf = LinearRegression()
clf.fit(X,y)

print(y)
print(clf.predict(X))

