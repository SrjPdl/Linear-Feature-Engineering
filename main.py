from model import Model
from utils import load_data, process_data, get_config, write_predictions_to_file

def main():
    
    config = get_config("./configs/config.yaml")
    train_data = load_data(config["train_file_path"])
    test_data = load_data(config["test_file_path"])
    X, y = process_data(train_data)

    degrees = range(1, config['max_degree'] + 1)
    base_model = Model(X,y, config["k_folds"])

    PLOT_STATE = "initial"

    predictions, best_degree, best_model = base_model.train(degrees, PLOT_STATE)

    best_feature_indices = base_model.select_best_features(best_degree, 5)

    X_new = X[:,best_feature_indices]
    new_model = Model(X_new, y, config["k_folds"])
    PLOT_STATE = "selected_5"

    print("Train MSE with all features: ", sum((y - predictions)**2)/len(y))
    print("Best model degree with all 8 features:", best_degree, "\n")

    predictions, best_degree, best_model = new_model.train(degrees, PLOT_STATE)
    X_test_selected = test_data.values[:,best_feature_indices]
    test_predictions = new_model.get_predictions(X_test_selected, (best_degree, best_model))
    write_predictions_to_file(test_predictions, config["prediction_file_path"])

    PLOT_STATE = "selected_4"
    new_4_model = Model(X_new[:,:-2], y, config["k_folds"])
    predictions, best_degree, best_model = new_4_model.train(degrees, PLOT_STATE)
    
    print(f"Train MSE with only {best_feature_indices} features:", sum((y - predictions)**2)/len(y))
    print(f"Best model degree with only {best_feature_indices} features:", best_degree)

if __name__ == '__main__':
    main()
