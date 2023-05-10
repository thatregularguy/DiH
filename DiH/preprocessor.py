from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.is_fit = False

    def fit(self, X):
        # Fit the scaler and imputer to the data
        self.scaler.fit(X)
        self.imputer.fit(self.scaler.transform(X))
        self.is_fit = True

    def transform(self, X):
        # Transform the data using the fitted scaler and imputer
        X_scaled = self.scaler.transform(X)
        X_imputed = self.imputer.transform(X_scaled)
        return X_imputed