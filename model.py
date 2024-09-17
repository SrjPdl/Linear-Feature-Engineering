import numpy as np
from least_squares import least_squares
from estimate_errors_kfold import estimate_errors_kfold
from utils import plot_p_vs_R, create_polynomial_features, fit_polynomial_regression, plot_features_vs_errors

class Model:
    def __init__(self, X, y, k_folds) -> None:
        self.X = X
        self.y = y
        self.k_folds = k_folds

    def __fit_model(self, degree):
        X_poly = create_polynomial_features(self.X, degree)
        beta = least_squares(X_poly, self.y)
        return beta
    
    def get_predictions(self, X, best_model):
        degree, beta = best_model
        X_poly = create_polynomial_features(X, degree)
        return X_poly @ beta
    
    def create_models(self, degrees):
        models = {}
        for degree in degrees:
            beta, X_poly = fit_polynomial_regression(self.X, self.y, degree)
            models[degree] = (beta, X_poly)
        return models
    
    def select_best_features(self, degree, num_features):
        n_features = self.X.shape[1]
        errors = []

        for i in range(n_features):
            X_subset = self.X[:, i].reshape(-1, 1)
            _, X_train_poly = fit_polynomial_regression(X_subset, self.y, degree)

            beta = least_squares(X_train_poly, self.y)
            y_pred = X_train_poly @ beta
            error = np.mean((self.y - y_pred) ** 2)
            errors.append((i, error))

        errors.sort(key=lambda x: x[1])
        best_feature_indices = [x[0] for x in errors[:num_features]]
        plot_features_vs_errors(errors, best_feature_indices)
        return best_feature_indices
    
    def select_poly_model(self, degrees, plot_state=""):
        errors = []
        for degree in degrees:
            train_errors, val_errors = estimate_errors_kfold(self.X, self.y, degree, self.k_folds, plot_state)
            errors.append((degree, train_errors, val_errors))
        
        plot_p_vs_R(errors, plot_state)
        best_degree = min(errors, key=lambda x: x[2])[0]

        return best_degree, self.__fit_model(best_degree)
    
    def train(self, degrees, plot_state=""):
        best_degree , best_model = self.select_poly_model(degrees, plot_state)
        predictions = self.get_predictions(self.X, [best_degree, best_model])
        return predictions, best_degree, best_model
