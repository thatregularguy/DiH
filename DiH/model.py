from sklearn.linear_model import LogisticRegression

class Model:
    def __init__(self, C=1, max_iter=1000, penalty='l2', solver='lbfgs', class_weight={0: 1, 1: 5}):
        self.model = LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver,
                                        class_weight=class_weight)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
