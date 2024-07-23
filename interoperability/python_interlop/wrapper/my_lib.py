import matplotlib.pyplot as plt
import numpy as np
import ctypes
from interoperability.python_interlop.wrapper.import_lib import init_lib

my_lib = init_lib()


class MyModel:
    def __init__(self, model_type: str, size: (int, tuple), is_classification: bool = False, is_3_classes: bool = False,
                 cluster_size: int = 0, gamma: float = 0, kernel:int = 1, kernel_value:float = 1.0,):
        if model_type not in ["ml", "mlp", "rbf", "svm"]:
            raise ValueError(f"Invalid model type: {model_type}")
        if not isinstance(size, (int, tuple)):
            raise ValueError("Size must be an integer or a list of integers")
        if cluster_size < 0:
            raise ValueError("Cluster size must be non-negative")
        if gamma < 0:
            raise ValueError("Gamma must be non-negative")

        self.train_data = []
        self.test_data = []
        self.__type = model_type
        self.__dims = size
        self.__is_classification = is_classification
        self.__cluster_size = cluster_size
        self.__gamma = gamma
        self.kernel = kernel
        self.kernel_value = kernel_value
        self.epsilon = 0

        if is_classification:
            self.__is_3_classes = is_3_classes
        else:
            self.__is_3_classes = False

        print("Initializing the model...")

        if not self.__is_3_classes or self.__type not in ["ml", "rbf", "svm"]:
            self.model = self._init_model()
        else:
            # Initialise 3 modèles pour comparer les résultats de chaque One vs Rest
            self.model = [self._init_model() for _ in range(3)]

    def _init_model(self):
        """
        Initializes a model using the rust lib
        :return: the address of the model
        """
        if self.__type == "ml":
            return my_lib.init_linear_model(self.__dims, self.__is_classification, self.__is_3_classes)
        elif self.__type == "mlp":
            raw_size = np.ctypeslib.as_ctypes(np.array(self.__dims, dtype=ctypes.c_uint32))
            return my_lib.init_mlp(raw_size, len(self.__dims), self.__is_classification, self.__is_3_classes)
        elif self.__type == "rbf":
            return my_lib.init_rbf(self.__dims, self.__cluster_size, self.__gamma)
        elif self.__type == "svm":
            return my_lib.init_svm(self.__dims, self.kernel, self.kernel_value)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_test=None, y_test=None,
              learning_rate=0.01, epochs=10_000, log_filename="model", model_filename="model",
              display_loss=False, display_tensorboard=False, save_model=False, sample_count=3, epsilon:float = 1e-3):
        """
        Trains the model on the given data
        :param epsilon:
        :param save_model:
        :param display_tensorboard:
        :param display_loss:
        :param sample_count:
        :param model_filename:
        :param x_train: Input data
        :param x_test: Input data
        :param y_test: Label data
        :param y_train: Label data
        :param learning_rate: Learning rate
        :param epochs: Number of epochs
        :param log_filename: nom de ficher logs
        """
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have same length")

        if x_test is None or y_test is None:
            x_test = x_train
            y_test = y_train

        if len(x_test) != len(y_test):
            raise ValueError("x_test and y_test must have same length")

        print("Training the model...")

        self.epsilon = epsilon
        train_data_size = len(y_train)
        test_data_size = len(y_test)

        # print(f"Training RBF with data_size={train_data_size}, lr={learning_rate}, epochs={epochs}, sample_count={
        # sample_count}")

        if self.__is_3_classes:
            y_train[y_train == 0] = -1
            y_test[y_test == 0] = -1

        self.train_data = list(map(list, zip(*x_train)))
        self.train_data.append(y_train)

        self.test_data = list(map(list, zip(*x_test)))
        self.test_data.append(y_test)

        x_train_flat = x_train.flatten().astype(ctypes.c_float)
        x_train_flat_ptr = x_train_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        x_test_flat = x_test.flatten().astype(ctypes.c_float)
        x_test_flat_ptr = x_test_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if not self.__is_3_classes or self.__type not in ["ml", "rbf", "svm"]:
            y_train_flat_ptr = y_train.flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            y_test_flat_ptr = y_test.flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._train_model(x_train_flat_ptr, y_train_flat_ptr, train_data_size, x_test_flat_ptr, y_test_flat_ptr,
                              test_data_size, sample_count, learning_rate, epochs,
                              ctypes.c_char_p(log_filename.encode()),
                              ctypes.c_char_p(model_filename.encode()), display_loss, display_tensorboard, save_model)
        else:
            y_train = np.transpose(y_train)
            y_test = np.transpose(y_test)
            for i in range(3):
                y_train_flat_ptr = y_train[i].flatten().astype(ctypes.c_float).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float))
                y_test_flat_ptr = y_test[i].flatten().astype(ctypes.c_float).ctypes.data_as(
                    ctypes.POINTER(ctypes.c_float))

                self._train_model(x_train_flat_ptr, y_train_flat_ptr, train_data_size, x_test_flat_ptr,
                                  y_test_flat_ptr,
                                  test_data_size, sample_count, learning_rate, epochs,
                                  ctypes.c_char_p(log_filename.encode()),
                                  ctypes.c_char_p(model_filename.encode()), display_loss, display_tensorboard,
                                  save_model, index=i)

    def _train_model(self,
                     x_train_flat_ptr: np.ndarray, y_train_flat_ptr: np.ndarray, train_data_size: int,
                     x_test_flat_ptr: np.ndarray, y_test_flat_ptr: np.ndarray, test_data_size: int, sample_count: int,
                     learning_rate: float, epochs: int, log_filename, model_filename, display_loss, display_tensorboard,
                     save_model, index=None):
        """
        Calls the lib to train the model
        """
        try:
            if self.__type == "ml":
                my_lib.train_linear_model(self.model if index is None else self.model[index],
                                          x_train_flat_ptr, y_train_flat_ptr, train_data_size,
                                          x_test_flat_ptr, y_test_flat_ptr, test_data_size,
                                          learning_rate, epochs,
                                          log_filename,
                                          model_filename, display_loss, display_tensorboard, save_model
                                          )
            elif self.__type == "mlp":
                print("begin training the mlp")
                my_lib.train_mlp(self.model, x_train_flat_ptr, y_train_flat_ptr, train_data_size,
                                 x_test_flat_ptr, y_test_flat_ptr, test_data_size,
                                 learning_rate, epochs,
                                 log_filename,
                                 model_filename, display_loss, display_tensorboard, save_model)
                print("finish training the mlp")

            elif self.__type == "rbf":
                if self.__is_classification:
                    my_lib.train_rbf_rosenblatt(self.model if index is None else self.model[index]
                                                , x_train_flat_ptr, y_train_flat_ptr, epochs, learning_rate,
                                                self.__dims, sample_count)
                else:
                    my_lib.train_rbf_regression(self.model, x_train_flat_ptr, y_train_flat_ptr, self.__dims,
                                                sample_count)
            elif self.__type == "svm":
                # the variable learning_rate is used for the parameter c of train_svm
                my_lib.train_svm(self.model if index is None else self.model[index], x_train_flat_ptr, y_train_flat_ptr,
                                 train_data_size, learning_rate, self.epsilon, x_train_flat_ptr, y_train_flat_ptr,
                                 train_data_size,)
        except Exception as e:
            print(f"Training failed due to {e}")
            raise

    def print(self, start_x: int = None, start_y: int = None, end_x: int = None, end_y: int = None, step=None):
        if self.__is_classification:
            self.print_classification(start_x=start_x, start_y=start_y, step=step, end_x=end_x, end_y=end_y)
        else:
            self.print_regression(start=start_x, end=end_x)

    def print_classification(self, end_x, end_y, step, start_x=0, start_y=0):
        print("Printing the classification results...")
        background_points = np.mgrid[start_x:end_x:step, start_y:end_y:step].reshape(2, -1).T
        background_points = np.round(background_points, decimals=3)
        background_colors = np.array(list(map(self._get_prediction_color, background_points)))
        # np.apply_along_axis(self._get_prediction_color, 1, background_points)

        fig, ax = plt.subplots()
        fig.patch.set_bounds(-5, -5, 10, 10)
        plt.scatter(background_points[:, 0], background_points[:, 1], c=background_colors)
        self.print_dataset()
        plt.show()
        plt.clf()

    def print_dataset(self):
        print("Printing the training data...")
        if self.__is_classification:
            plt.scatter(self.train_data[0], self.train_data[1], c=self._get_train_colors())
        else:
            if self.__dims == 1 or (self.__type == "mlp" and self.__dims[0] == 1):
                plt.scatter(self.train_data[0], self.train_data[1])

    def _get_prediction_color(self, point: np.ndarray):
        prediction = self._predict_value(point)
        # print(prediction)
        if not self.__is_3_classes:
            return "lightblue" if prediction > 0 else "pink"
        else:
            result = np.argmax(prediction)
            return ["lightblue", "pink", "lightgreen"][result] if result in [0, 1, 2] else "white"

    def _get_train_colors(self):
        train_colors = []
        if not self.__is_3_classes:
            train_colors = ["blue" if result == 1 else "red" for result in self.train_data[2]]
        else:
            for point in self.train_data[2]:
                if point[0] == 1:
                    train_colors.append("blue")
                elif point[1] == 1:
                    train_colors.append("red")
                else:
                    train_colors.append("green")

        return train_colors

    def _predict_value(self, point: np.ndarray):
        point_pointer = np.ctypeslib.as_ctypes(np.array(point, dtype=ctypes.c_float))
        if self.__type == "ml":
            if not self.__is_3_classes:
                return my_lib.predict_linear_model(self.model, point_pointer)
            else:
                return [my_lib.predict_linear_model(self.model[i], point_pointer) for i in range(3)]
        elif self.__type == "mlp":
            if not self.__is_3_classes:
                return my_lib.predict_mlp(self.model, point_pointer)[0]
            else:
                return np.ctypeslib.as_array(my_lib.predict_mlp(self.model, point_pointer), (3,))
        elif self.__type == "rbf":
            if self.__is_classification:
                if not self.__is_3_classes:
                    return my_lib.predict_rbf_classification(self.model, point_pointer)
                else:
                    return [my_lib.predict_rbf_regression(self.model[i], point_pointer) for i in range(3)]
            else:
                return my_lib.predict_rbf_regression(self.model, point_pointer)
        elif self.__type == "svm":
            if not self.__is_3_classes:
                return my_lib.predict_svm(self.model, point_pointer)
            else:
                return [my_lib.predict_svm(self.model[i], point_pointer) for i in range(3)]

    def print_regression(self, start: float = 0, end: float = 3.2):
        x = [i * 0.1 for i in range(round(start * 10), round(end * 10))]
        y = [self._predict_value(np.array([v], dtype=ctypes.c_float)) for v in x]

        self.print_dataset()
        plt.plot(x, y)
        plt.show()

    def print_regression_3d(self, start: float = 0, end: float = 3.2):
        X = np.array(self.train_data)
        x_surf, y_surf = np.meshgrid(np.linspace(X[0].min(), X[0].max(), 100),
                                     np.linspace(X[1].min(), X[1].max(), 100))

        result = (np.array([self._predict_value(np.array([x, y]))
                            for x, y in zip(x_surf.ravel(), y_surf.ravel())]).reshape(x_surf.shape))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[0], X[1], X[2], c="blue")
        ax.plot_surface(x_surf, y_surf, result, color="grey", alpha=0.5)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Y')
        plt.show()

    def free_model(self):
        if hasattr(self, "model") and self.model is not None:
            print("Deleting last model...")
            if self.__type == "ml":
                if not self.__is_3_classes:
                    my_lib.free_linear_model(self.model)
                else:
                    for i in range(3):
                        my_lib.free_linear_model(self.model[i])
            elif self.__type == "mlp":
                my_lib.free_mlp(self.model)
            elif self.__type == "rbf":
                if not self.__is_3_classes:
                    my_lib.free_rbf(self.model)
                else:
                    for i in range(3):
                        my_lib.free_rbf(self.model[i])
            elif self.__type == "svm":
                if not self.__is_3_classes:
                    my_lib.free_svm(self.model)
                else:
                    for i in range(3):
                        my_lib.free_svm(self.model[i])
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_model()

    def __del__(self):
        self.free_model()
