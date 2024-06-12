import matplotlib.pyplot as plt
import numpy
import numpy as np
import ctypes as ctypes

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

# -------------------------- init mlp --------------------------
# pub extern "C" fn init_mlp(npl: *mut u32, npl_size: u32) -> *mut MultiLayerPerceptron
my_lib.init_mlp.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32, ctypes.c_bool]
my_lib.init_mlp.restype = ctypes.c_void_p

# pub extern "C" fn train_mlp(  model: *mut MultiLayerPerceptron, inputs: *mut c_float, outputs: *mut c_float, row: u32,
#                               alpha: c_float, nb_iteration: u32)
my_lib.train_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32]
my_lib.train_mlp.restype = None

# pub extern "C" fn predict_mlp(model: *mut MultiLayerPerceptron,sample_inputs: *mut f32,
#                               sample_inputs_size: usize,is_classification: bool,) -> *mut f32
my_lib.predict_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

# pub extern "C" fn free_mlp(model: *mut MultiLayerPerceptron)
my_lib.free_mlp.argtypes = [ctypes.c_void_p]
my_lib.free_mlp.restype = None

# ---------------------------- init RBF --------------------------
# pub extern "C" fn init_rbf(input_dim : i32, cluster_num : i32, gamma : f32) -> *mut RadicalBasisFunctionNetwork
my_lib.init_rbf.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
my_lib.init_rbf.restype = ctypes.c_void_p

# pub extern "C" fn train_rbf_regression(model : *mut RadicalBasisFunctionNetwork, sample_inputs_flat : *mut f32,
#                                           expected_outputs : *mut f32, inputs_size : i32, sample_count : i32)
my_lib.train_rbf_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int32, ctypes.c_int32]
my_lib.train_rbf_regression.restype = None

# pub extern "C" fn predict_rbf_regression(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32) -> f32
my_lib.predict_rbf_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
my_lib.predict_rbf_regression.restype = ctypes.c_float

# pub extern "C" fn train_rbf_rosenblatt(model : *mut RadicalBasisFunctionNetwork, sample_inputs_flat : *mut f32,
#           expected_outputs : *mut f32, iterations_count : i32, alpha : f32, inputs_size : i32, sample_count : i32)
my_lib.train_rbf_rosenblatt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                                        ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32]
my_lib.train_rbf_rosenblatt.restype = None

# pub extern "C" fn predict_rbf_classification(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32)-> f32
my_lib.predict_rbf_classification.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
my_lib.predict_rbf_classification.restype = ctypes.c_float

class MyModel:
    def __init__(self, model_type, size, is_classification=False, is_3_classes=False, cluster_size=0, gamma=0, ):
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
            elif self.__type == "mlp":
                raw_size = np.ctypeslib.as_ctypes(np.array(size, dtype=ctypes.c_uint32))
                self.model = my_lib.init_mlp(raw_size, len(size), is_classification)
            elif self.__type == "rbf":
                self.cluster_size = cluster_size
                self.gamma = gamma
                self.model = my_lib.init_rbf(size, cluster_size, gamma)
        else:
            self.model = []
            if self.__type == "ml":
                for i in range(3):
                    self.model.append(my_lib.init_linear_model(self.__dims, False))
            elif self.__type == "mlp":
                raw_size = np.ctypeslib.as_ctypes(np.array(size, dtype=ctypes.c_uint32))
                for i in range(3):
                    self.model.append(my_lib.init_mlp(raw_size, len(size), is_classification))

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
            elif self.__type == "mlp":
                my_lib.train_mlp(self.model, x_flat_ptr, y_flat_ptr, data_size, learning_rate, epochs)
            elif self.__type == "rbf":
                if self.__is_classification:
                    my_lib.train_rbf_rosenblatt(self.model, x_flat_ptr, y_flat_ptr, data_size, learning_rate, epochs)
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
                    elif self.__type == "mlp":
                        prediction = my_lib.predict_mlp(self.model, points_pointer)[0]
                        if prediction > 0:
                            prediction = 1
                        else:
                            prediction = -1
                    else:
                        prediction = 0

                    if prediction == 1:
                        background_colors.append("lightblue")
                    elif prediction == -1:
                        background_colors.append("pink")
                    else:
                        print("Problème de prédiction :", prediction)
                else:
                    prediction = []
                    if self.__type == "ml":
                        for i in range(3):
                            prediction.append(my_lib.predict_linear_model(self.model[i], points_pointer))
                    elif self.__type == "mlp":
                        for i in range(3):
                            prediction.append(my_lib.predict_mlp(self.model[i], points_pointer))

                    result = np.argmax(prediction)
                    if result == 0:
                        background_colors.append("lightblue")
                    elif result == 1:
                        background_colors.append("pink")
                    elif result == 2:
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

                pos_y = round(pos_y + step, 4)  # round to avoid floating point drift
            pos_x = round(pos_x + step, 4)

        background_points = np.array(background_points)

        fig, ax = plt.subplots()

        fig.patch.set_bounds(-5, -5, 10, 10)
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
        plt.clf()

    def print_regression(self, start=0, size=3.2):
        x = [i * 0.1 for i in range(32)]
        y = []

        if self.__dims == 1 or (self.__type == "mlp" and self.__dims[0] == 1):
            if self.__type == "ml":
                for v in x:
                    point = np.array([v], dtype=ctypes.c_float)
                    points_pointer = np.ctypeslib.as_ctypes(point)
                    y.append(my_lib.predict_linear_model(self.model, points_pointer))
            elif self.__type == "mlp":
                for v in x:
                    point = np.array([v], dtype=ctypes.c_float)
                    points_pointer = np.ctypeslib.as_ctypes(point)
                    y.append(my_lib.predict_mlp(self.model, points_pointer)[0])

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

    def delete(self):
        if self.__type == "ml":
            my_lib.free_linear_model(self.model)
        elif self.__type == "mlp":
            my_lib.free_mlp(self.model)
