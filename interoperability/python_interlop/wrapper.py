import matplotlib.pyplot as plt
import numpy
import numpy as np
import ctypes as ctypes

lib_path = "../../mylib/target/release/mylib.dll"
my_lib = ctypes.cdll.LoadLibrary(lib_path)

# --------------------------init_linear_model--------------------------
# pub extern "C" fn init_linear_model(input_count: u32) -> *mut LinearModel;
my_lib.init_linear_model.argtypes = [ctypes.c_uint32]
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


class MyModel:
    def __init__(self, model_type, size):
        self.train_data = ()
        if model_type == "ml":
            self.model = my_lib.init_linear_model(size)
            self.__type = model_type

    def train(self, x: numpy.array, y: numpy.array, learning_rate: float, epochs: int):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if self.__type not in ["ml"]:
            raise ValueError("incorrect model type")

        X_flat = x.flatten().astype(ctypes.c_float)
        Y_flat = y.flatten().astype(ctypes.c_float)
        X_flat_ptr = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        Y_flat_ptr = Y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        data_size = len(y)

        self.train_data = ([i[0] for i in x], [i[1] for i in x], y)

        if self.__type == "ml":
            my_lib.train_linear_model(self.model, X_flat_ptr, Y_flat_ptr, data_size, learning_rate, epochs)

    def print_predictions(self, size_x, size_y, step):
        background_points = []
        background_colors = []
        x1 = 0
        while x1 < size_x:
            x2 = 0
            while x2 < size_y:
                background_points.append([x1, x2])
                points_pointer = np.ctypeslib.as_ctypes(np.array([x1, x2], dtype=ctypes.c_float))

                prediction = 0
                if self.__type == "ml":
                    prediction = my_lib.predict_linear_model(self.model, points_pointer)

                if prediction >= 0:
                    background_colors.append("lightblue")
                else:
                    background_colors.append("pink")

                x2 += step
            x1 += step

        background_points = np.array(background_points)
        plt.scatter(background_points[:, 0], background_points[:, 1], c=background_colors)

        train_colors = []
        for result in self.train_data[2]:
            if result >= 0:
                train_colors.append("blue")
            else:
                train_colors.append("red")

        plt.scatter(self.train_data[0], self.train_data[1], c=train_colors)

        plt.show()

    def __del__(self):
        if self.__type != "ml":
            my_lib.free_linear_model(self.model)
