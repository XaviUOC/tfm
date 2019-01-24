#!/usr/bin/env python
import numpy as np
import pandas as pd
import functools
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from matplotlib import pyplot as plt
from keras import backend as K
from mining import compute_trend


def load_dataset(filename):
    df = pd.read_csv(filename)
    df.timestamp = pd.to_datetime(df.timestamp)
    return df.set_index("timestamp")


def normalize(values):
    # Convert to float
    values = values.astype(np.float)

    # Normalize
    scaler = MinMaxScaler((0, 1))
    return scaler.fit_transform(values)


def train_test(data, proportion=2/3):
    # Split in train/test
    mask = np.random.rand(len(data)) < proportion
    train, test = data[mask], data[~mask]

    # The last dimension is Y, the rest X
    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    return train_x, train_y, test_x, test_y


def evalute_model(trend, prediction):
    # return 100 * (trend & prediction).sum() / len(trend)
    return 100 * precision_score(trend, prediction)


def print_metric(model_name, trend, prediction):
    metric = evalute_model(trend, prediction)
    print("{0} precision: {1:3.2f} %".format(model_name, metric))
    print("Confusion matrix:")
    print(confusion_matrix(trend, prediction))


def minimum_viable_algorithm(df):
    # The prediction is what happened the last 3 hours in average
    avg_high = (df.high_1 + df.high_2 + df.high_3) / 3.
    prediction = compute_trend(df.close, avg_high)
    print_metric("Finger rule", df.trend.values, prediction.values)


def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


# Keras precision metric
precision = as_keras_metric(tf.metrics.precision)


def logistic_regression(train_x, train_y, test_x, test_y, epochs):
    # Calculate the ratio of the imbalanced datasets
    ratio = np.sum(train_x == 0) / np.sum(train_x == 1)

    train_y = train_y.astype(np.int)
    test_y = test_y.astype(np.int)

    # Implement the model
    model = Sequential()
    # model.add(Dense(1, input_dim=train_x.shape[1]))
    # model.add(Activation("softmax"))
    model.add(Dense(train_x.shape[1], input_dim=train_x.shape[1],
              activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam",
                  metrics=[precision])
    model.fit(train_x, train_y, epochs=epochs, class_weight={0: 1., 1: ratio})

    # Check the goodness of the prediction
    prediction = model.predict_classes(test_x)
    prediction = prediction.reshape(prediction.shape[0])

    # Print the result
    print_metric("Logistic regresion", test_y, prediction)


def lstm(train_x, train_y, test_x, test_y, period, features, epochs):
    ratio = np.sum(train_x == 0) / np.sum(train_x == 1)

    # It's needed reshape input to be 3D [samples, timeframe, features]
    train_x = train_x.reshape((train_x.shape[0], period, features))
    test_x = test_x.reshape((test_x.shape[0], period, features))

    # Implement the model
    model = Sequential()
    model.add(LSTM(256, input_shape=(train_x.shape[1:])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # model.add(Activation('softmax'))
    # model.compile(loss='mae', optimizer='sgd')

    # from keras.utils import to_categorical
    # train_y = to_categorical(train_y)

    model.compile(loss="mae", optimizer="sgd")
    history = model.fit(train_x, train_y, epochs=epochs, batch_size=1,
                        class_weight={0: 1., 1: ratio})

    # Check the goodness of the prediction
    prediction = model.predict_classes(test_x)
    prediction = prediction.reshape(prediction.shape[0])

    # Print the result
    print_metric("LSTM regresion", test_y, prediction)
    pass
    # evaluation = model.evaluate(test_x, test_y)
    # print("Summary: Loss over the test dataset: {0:.2}, Accuracy: {1:.2}".format(*evaluation))

    # Ploting
    plt.plot(history.history["loss"], label="train")
    plt.legend()
    plt.savefig("lstm.png")


def main(filename):
    # Initialize random seed
    np.random.seed(666)

    # Load dataset
    df = load_dataset(filename)

    # Model 1: Finger rule
    minimum_viable_algorithm(df)

    # Normalize data
    norm_data = normalize(df.values)

    # split the data in training & test sets
    train_x, train_y, test_x, test_y = train_test(norm_data)
    #x, y = norm_data[:, :-1], norm_data[:, -1:]
    # TODO: train_x, train_y, test_x, test_y = train_test_split(x, y, test_size=.33)

    # Model 2: Logistic classification
    # logistic_regression(train_x, train_y, test_x, test_y, epochs=50)

    # Model 3: LSTM
    period, features = 3, 10
    lstm(train_x, train_y, test_x, test_y, period + 1, features, epochs=200)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: {} <file>".format(sys.argv[0]))
        sys.exit(1)

    main(sys.argv[1])
