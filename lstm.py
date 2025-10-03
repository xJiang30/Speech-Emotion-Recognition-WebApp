from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import callbacks
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import datetime
import config


#load config
config = config.load_config("config.yaml")

def train_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=config["validation_split"], random_state=42)

    model = Sequential([
        LSTM(520, return_sequences=True, input_shape=X.shape[1:3]),
        LSTM(347, return_sequences=True),
        LSTM(213, return_sequences=True),
        LSTM(192),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(Y.shape[1], activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    print(model.summary())

    checkpoint_path = config["model_path"]
    mcp_save = callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_categorical_accuracy', mode='max')
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.005, patience=100)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(
        X_train, y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[mcp_save, rlrop, tensorboard_callback],
        use_multiprocessing=True
    )

    return model, history, X_test, y_test

def evaluate_model(enc, model_path, X_test, y_test, dataset):
    model = load_model(model_path)
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    y_pred = model.predict(X_test)
    y_test_labels = enc.inverse_transform(y_test)
    y_pred_labels = enc.inverse_transform(y_pred)

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset['Emotions'].unique())
    fig, ax = plt.subplots(figsize=(12, 12), tight_layout=True)
    disp.plot(cmap=plt.cm.Greens, ax=ax)
    plt.show()

    print(classification_report(y_test_labels, y_pred_labels))


