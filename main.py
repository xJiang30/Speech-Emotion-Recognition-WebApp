import pandas as pd
import config
import process_and_feature as feat
import lstm
import plot_utils

#load config
config = config.load_config("config.yaml")

# read dataset
dataset = pd.read_csv(config["df_path"])

# pre-processing & extract feature
X,Y,enc = feat.build_feature_set(dataset)

# train
model, history, X_test, y_test = lstm.train_model(X, Y)

# evaluate
lstm.evaluate_model(enc, config["model_path"],X_test, y_test,dataset)

# plot
plot_utils.plot_training_history(history)