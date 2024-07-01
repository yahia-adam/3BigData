import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_logdir = "./logs"

    seeds = [20, 17, 7, 3, 28]
    batch_size = 1024
    epochs = 250
    lr = 0.05

    for seed in seeds:
        experiment_name = f"std_ds_seed_{seed}_experiment_bs_{batch_size}_lr{lr}_3_512_256_128_hl"

        final_logdir = os.path.join(base_logdir, experiment_name)

        keras.utils.set_random_seed(seed)

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        print(x_train.shape)
        print(x_test.shape)

        x_train_mean = np.mean(x_train)
        x_train_std = np.std(x_train)

        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        print(y_train[:5])

        y_train_vector = np.ones((len(y_train), 10)) * -1.0
        for (i, label) in enumerate(y_train):
            y_train_vector[i][label] = 1.0

        y_test_vector = np.ones((len(y_test), 10)) * -1.0
        for (i, label) in enumerate(y_test):
            y_test_vector[i][label] = 1.0

        print(y_train_vector[:5])

        # plt.imshow(x_train[1])
        # plt.show()

        model = keras.models.Sequential()
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(512, activation=keras.activations.tanh))
        model.add(keras.layers.Dense(256, activation=keras.activations.tanh))
        model.add(keras.layers.Dense(128, activation=keras.activations.tanh))
        model.add(keras.layers.Dense(10, activation=keras.activations.tanh))

        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.SGD(learning_rate=lr)
        )

        model.fit(
            x_train,
            y_train_vector,
            validation_data=(x_test, y_test_vector),
            callbacks=[
                keras.callbacks.TensorBoard(final_logdir)
            ],
            epochs=epochs,
            batch_size=batch_size,
        )