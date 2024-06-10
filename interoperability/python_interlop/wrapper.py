import matplotlib.pyplot as plt
import numpy
import numpy as np
import ctypes as ctypes
from mpl_toolkits.mplot3d import Axes3D

lib_path = "../../mylib/target/release/mylib.dll"
my_lib = ctypes.cdll.LoadLibrary(lib_path)

# --------------------------init_linear_model--------------------------
# pub extern "C" fn init_linear_model(input_count: u32) -> *mut LinearModel;
my_lib.init_linear_model.argtypes = [ctypes.c_uint32, ctypes.c_bool]
my_lib.init_linear_model.restype = ctypes.c_void_p

# pub extern "C" fn train_linear_model( model: *mut LinearModel, features: *const c_float,
#                                       labels: *const c_float, data_size: u32, learning_rate: f32, epochs: u32)
my_lib.train_linear_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                      ctypes.POINTER(ctypes.c_float),
                                      ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32]
my_lib.train_linear_model.restype = None

# pub extern "C" fn predict_linear_model(model: *mut LinearModel, inputs: *mut f32) -> c_float
my_lib.predict_linear_model.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
my_lib.predict_linear_model.restype = ctypes.c_float

# pub extern "C" fn free_linear_model(model: *mut LinearModel)
my_lib.free_linear_model.argtypes = [ctypes.c_void_p]
my_lib.free_linear_model.restype = None


## -------------------------- init mlp --------------------------
# pub extern "C" fn init_mlp(npl: *mut u32, npl_size: u32) -> *mut MultiLayerPerceptron
my_lib.init_mlp.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
my_lib.init_mlp.restype = ctypes.c_void_p

# pub extern "C" fn train_mlp(  model: *mut MultiLayerPerceptron, inputs: *mut c_float, outputs: *mut c_float, row: u32,
#                               alpha: c_float, nb_iteration: u32, is_classification: bool)
my_lib.train_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_bool]
my_lib.train_mlp.restype = None

# pub extern "C" fn predict_mlp(model: *mut MultiLayerPerceptron,sample_inputs: *mut f32,
#                               sample_inputs_size: usize,is_classification: bool,) -> *mut f32
my_lib.predict_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

# pub extern "C" fn free_mlp(model: *mut MultiLayerPerceptron)
my_lib.free_mlp.argtypes = [ctypes.c_void_p]
my_lib.free_mlp.restype = None


class MyModel:
    def __init__(self, model_type, size, is_classification=False, is_3_classes=False):
        self.train_data = []
        self.__type = model_type
        self.__dims = size
        self.__is_classification = is_classification
        if is_classification:
            self.__is_3_classes = is_3_classes
        else:
            self.__is_3_classes = False

        if not is_3_classes:
            if self.__type == "ml":
                self.model = my_lib.init_linear_model(self.__dims, is_classification)
        else:
            if self.__type == "ml":
                self.model = []
                for i in range(3):
                    self.model.append(my_lib.init_linear_model(self.__dims, is_classification))

        # if model_type == "ml":
        #     self.models = []
        #     for i in range(classes):
        #         self.models.append(my_lib.init_linear_model(size))
        # elif model_type == "mlp":
        #     size_ptr = np.ctypeslib.as_ctypes(
        #         np.array(size, dtype=ctypes.c_uint32)
        #     )
        #     self.model = my_lib.init_mlp(size_ptr)
        #
        # if classes != 2:
        #     self.__classes += 1

    def train(self, x: numpy.array, y: numpy.array, learning_rate: float, epochs: int):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if self.__type not in ["ml"]:
            raise ValueError("incorrect model type")

        data_size = len(y)

        if self.__is_3_classes:
            y[y == 0] = -1

        self.train_data = list(map(list, zip(*x)))
        self.train_data.append(y)

        x_flat = x.flatten().astype(ctypes.c_float)
        x_flat_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if not self.__is_3_classes:
            y_flat_ptr = y.flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if self.__type == "ml":
                my_lib.train_linear_model(self.model, x_flat_ptr, y_flat_ptr, data_size, learning_rate, epochs)
        else:
            y = np.transpose(y)

            for i in range(3):
                y_flat_ptr = y[i].flatten().astype(ctypes.c_float).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                if self.__type == "ml":
                    my_lib.train_linear_model(self.model[i], x_flat_ptr, y_flat_ptr, self.__dims, learning_rate, epochs)

    def print_classif(self, size_x, size_y, step, start_x=0, start_y=0):
        background_points = []
        background_colors = []

        pos_x = start_x
        while pos_x < size_x:
            pos_y = start_y
            while pos_y < size_y:
                background_points.append([pos_x, pos_y])
                points_pointer = np.ctypeslib.as_ctypes(np.array([pos_x, pos_y], dtype=ctypes.c_float))

                if not self.__is_3_classes:
                    if self.__type == "ml":
                        prediction = my_lib.predict_linear_model(self.model, points_pointer)
                    else:
                        prediction = 0

                    if prediction == 1:
                        background_colors.append("lightblue")
                    else:
                        background_colors.append("pink")
                else:
                    prediction = []
                    if self.__type == "ml":
                        for i in range(3):
                            prediction.append(my_lib.predict_linear_model(self.model[i], points_pointer))
                    if prediction[0] == [1.0, -1.0, -1.0]:
                        background_colors.append("lightblue")
                    elif prediction[1] == [-1.0, 1.0, -1.0]:
                        background_colors.append("pink")
                    elif prediction[2] == [-1.0, -1.0, 1.0]:
                        background_colors.append("lightgreen")
                    else:
                        background_colors.append("white")

                # if self.__type == "ml":
                #     prediction = []
                #     for i in range(self.__classes - 1):
                #         prediction.append(my_lib.predict_linear_model(self.models[i], points_pointer))
                #
                #     prediction = np.array(prediction)
                #

                pos_y += step
            pos_x += step

        background_points = np.array(background_points)
        plt.scatter(background_points[:, 0], background_points[:, 1], c=background_colors)

        train_colors = []
        if not self.__is_3_classes:
            for result in self.train_data[2]:
                if result == 1:
                    train_colors.append("blue")
                else:
                    train_colors.append("red")
        else:
            for point in self.train_data[2]:
                if point[0] == 1:
                    train_colors.append("blue")
                elif point[1] == 1:
                    train_colors.append("red")
                else:
                    train_colors.append("green")

        plt.scatter(self.train_data[0], self.train_data[1], c=train_colors)
        plt.show()

    def print_regression(self, start=0, size=1):
        x = []
        y = []
        if self.__type == "ml":
            if self.__dims == 1:
                x = [0.0, 3.0]
                for v in x:
                    point = np.array([v], dtype=ctypes.c_float)
                    points_pointer = np.ctypeslib.as_ctypes(point)
                    y.append(my_lib.predict_linear_model(self.model, points_pointer))

                plt.scatter(self.train_data[0], self.train_data[1])
                plt.plot(x, y)

                plt.show()
            elif self.__dims == 2:
                x = [[0.0, 0.0], [3.0, 3.0]]
                for v in x:

                    points_pointer = np.ctypeslib.as_ctypes(np.array(v, dtype=ctypes.c_float))
                    y.append(my_lib.predict_linear_model(self.model, points_pointer))

                ax = plt.figure().add_subplot(111, projection='3d')

                ax.scatter(self.train_data[0], self.train_data[1], self.train_data[2])
                ax.plot_surface(x[0], x[1], y)

                plt.show()

    def __del__(self):
        if self.__type != "ml":
            my_lib.free_linear_model(self.model)
