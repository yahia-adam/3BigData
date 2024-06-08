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
    def __init__(self, model_type, size, classes=2):
        self.classes = classes
        self.train_data = ()
        self.__type = model_type

        if model_type == "ml":
            self.models = []
            for i in range(classes):
                self.models.append(my_lib.init_linear_model(size))

        elif model_type == "mlp":
            size_ptr = np.ctypeslib.as_ctypes(
                np.array(size, dtype=ctypes.c_uint32)
            )
            self.model = my_lib.init_mlp(size_ptr)

        if classes != 2:
            self.classes += 1

    def train(self, x: numpy.array, y: numpy.array, learning_rate: float, epochs: int):
        if len(x) != len(y):
            raise ValueError("x and y must have same length")
        if self.__type not in ["ml"]:
            raise ValueError("incorrect model type")

        if self.classes == 2:
            y = np.array([[1, 0] if value == 1 else [0, 1] for value in y])
        y[:][y == 0] = -1
        data_size = len(y)

        self.train_data = ([i[0] for i in x], [i[1] for i in x], y)
        y = y.transpose()

        for i in range(self.classes - 1):
            x_flat = x.flatten().astype(ctypes.c_float)
            y_flat = y[i].flatten().astype(ctypes.c_float)
            x_flat_ptr = x_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            y_flat_ptr = y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

            if self.__type == "ml":
                my_lib.train_linear_model(self.models[i], x_flat_ptr, y_flat_ptr, data_size, learning_rate, epochs)

    def print_classif(self, size_x, size_y, step, start_x=0, start_y=0):
        background_points = []
        background_colors = []

        pos_x = start_x
        while pos_x < size_x:
            pos_y = start_y
            while pos_y < size_y:
                background_points.append([pos_x, pos_y])
                points_pointer = np.ctypeslib.as_ctypes(np.array([pos_x, pos_y], dtype=ctypes.c_float))

                prediction = 0
                if self.__type == "ml":
                    prediction = []
                    for i in range(self.classes - 1):
                        prediction.append(my_lib.predict_linear_model(self.models[i], points_pointer))

                    prediction = np.array(prediction)

                    if prediction[0] == 1:
                        background_colors.append("lightblue")
                    elif self.classes == 2 or prediction[1] == 1:
                        background_colors.append("pink")
                    else:
                        background_colors.append("lightgreen")

                pos_y += step
            pos_x += step
        background_points = np.array(background_points)
        plt.scatter(background_points[:, 0], background_points[:, 1], c=background_colors)

        train_colors = []
        for result in self.train_data[2]:
            result = np.array(result)
            if np.all(result == result[0]):
                train_colors.append("white")
            elif np.argmax(result) == 0:
                train_colors.append("blue")
            elif np.argmax(result) == 1:
                train_colors.append("red")
            else:
                train_colors.append("green")

        plt.scatter(self.train_data[0], self.train_data[1], c=train_colors)

        plt.show()

    def __del__(self):
        if self.__type != "ml":
            my_lib.free_linear_model(self.model)
