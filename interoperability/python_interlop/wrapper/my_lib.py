import matplotlib.pyplot as plt
import numpy as np
import ctypes
from interoperability.python_interlop.wrapper.import_lib import init_lib

my_lib = init_lib()


class MyModel:
    def __init__(self, model_type: str, size: (int, tuple), is_classification: bool = False, is_3_classes: bool = False,
                 cluster_size: int = 0, gamma: float = 0, ):
        if model_type not in ["ml", "mlp", "rbf"]:
            raise ValueError(f"Invalid model type: {model_type}")
        if not isinstance(size, (int, tuple)):
            raise ValueError("Size must be an integer or a list of integers")
        if cluster_size < 0:
            raise ValueError("Cluster size must be non-negative")
        if gamma < 0:
            raise ValueError("Gamma must be non-negative")

        self.train_data = []
        self.__type = model_type
        self.__dims = size
        self.__is_classification = is_classification
        self.__cluster_size = cluster_size
        self.__gamma = gamma

        if is_classification:
            self.__is_3_classes = is_3_classes
        else:
            self.__is_3_classes = False

        print("Initializing the model...")

        if not self.__is_3_classes or self.__type != "ml":
            self.model = self._init_model()
        else:
            # Initialise les modèles linéaires en tant que regression pour pouvoir
            # comparer les résultat de chaque One vs Rest
            self.__is_classification = False
            self.model = [self._init_model() for _ in range(3)]
            self.__is_classification = True

    def _init_model(self):
        """
        Initializes a model using the rust lib
        :return: the address of the model
        """
        if self.__type == "ml":
            return my_lib.init_linear_model(self.__dims, self.__is_classification)
        elif self.__type == "mlp":
            raw_size = np.ctypeslib.as_ctypes(np.array(self.__dims, dtype=ctypes.c_uint32))
            return my_lib.init_mlp(raw_size, len(self.__dims), self.__is_classification)
        elif self.__type == "rbf":
            return my_lib.init_rbf(self.__dims, self.__cluster_size, self.__gamma)

    def train(self, x: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int):
        """
        Trains the model on the given data
        :param x: Input data
        :param y: Label data
        :param learning_rate: Learning rate
        :param epochs: Number of epochs
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        print("Training the model...")

        data_size = len(y)
        sample_count = len(x)

        if self.__is_3_classes:
            y[y == 0] = -1

        self.train_data = list(map(list, zip(*x)))
        self.train_data.append(y)

        x_flat = x.flatten().astype(ctypes.c_float)
        x_flat_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if not self.__is_3_classes or self.__type != "ml":
            y_flat_ptr = y.flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._train_model(x_flat_ptr, y_flat_ptr, data_size, sample_count, learning_rate, epochs)
        else:
            y = np.transpose(y)
            for i in range(3):
                y_flat_ptr = y[i].flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                if self.__type == "ml":
                    self._train_model(x_flat_ptr, y_flat_ptr, data_size, sample_count, learning_rate, epochs, i)

    def _train_model(self, x_flat_ptr: np.ndarray, y_flat_ptr: np.ndarray, data_size: int, sample_count: int,
                     learning_rate: float, epochs: int, index=None):
        """
        Calls the lib to train the model
        """
        try:
            if self.__type == "ml":
                my_lib.train_linear_model(self.model if index is None else self.model[index], x_flat_ptr, y_flat_ptr,
                                          data_size, learning_rate, epochs)
            elif self.__type == "mlp":
                my_lib.train_mlp(self.model, x_flat_ptr, y_flat_ptr, data_size, learning_rate, epochs)
            elif self.__type == "rbf":
                if self.__is_classification:
                    my_lib.train_rbf_rosenblatt(self.model, x_flat_ptr, y_flat_ptr, data_size, learning_rate,
                                                epochs, sample_count)
                # else:
                #     my_lib.train_rbf_regression(self.model, x_flat_ptr, y_flat_ptr, inputs_size, sample_count)
        except Exception as e:
            print(f"Training failed due to {e}")
            raise

    def print(self, start_x: int = None, start_y: int = None, end_x: int = None, end_y: int = None, step=None):
        if self.__is_classification:
            self.print_classification(start_x=start_x, start_y=start_y, step=step, end_x=end_x, end_y=end_y)
        else:
            self.print_regression(start=start_x, end=end_x)

    def print_classification(self, end_x, end_y, step, start_x=0, start_y=0):
        background_points = np.mgrid[start_x:end_x:step, start_y:end_y:step].reshape(2, -1).T
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
            return my_lib.predict_mlp(self.model, point_pointer)[0]
        elif self.__type == "rbf":
            if self.__is_classification:
                return my_lib.predict_rbf_classification(self.model, point_pointer)
            else:
                return my_lib.predict_rbf_regression(self.model, point_pointer)

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
                my_lib.free_rbf(self.model)
            self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free_model()

    def __del__(self):
        self.free_model()
