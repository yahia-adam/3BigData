import matplotlib.pyplot as plt
import numpy as np
import ctypes as ctypes

lib_path = r"./../../../../mylib/target/debug/mylib.dll"
my_lib = ctypes.cdll.LoadLibrary(lib_path)

## -------------------------- init mlp --------------------------
# pub extern "C" fn init_mlp(npl: *mut u32, npl_size: u32) -> *mut MultiLayerPerceptron
my_lib.init_mlp.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
my_lib.init_mlp.restype = ctypes.c_void_p

# pub extern "C" fn train_mlp( model: *mut MultiLayerPerceptron, inputs: *mut c_float, outputs: *mut c_float, row: u32, alpha: c_float, nb_iteration: u32, is_classification: bool)
my_lib.train_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                             ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_bool]
my_lib.train_mlp.restype = None

# pub extern "C" fn predict_mlp(model: *mut MultiLayerPerceptron,sample_inputs: *mut f32,sample_inputs_size: usize,is_classification: bool,) -> *mut f32
my_lib.predict_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

# pub extern "C" fn free_mlp(model: *mut MultiLayerPerceptron)
my_lib.free_mlp.argtypes = [ctypes.c_void_p]
my_lib.free_mlp.restype = None

X = np.array([
    [1, 1],
    [2, 3],
    [3, 3]
])
Y = np.array([
    1,
    -1,
    -1
])

X_flat = X.flatten().astype(ctypes.c_float)
Y_flat = Y.flatten().astype(ctypes.c_float)
X_flat_ptr = X_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
Y_flat_ptr = Y_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
data_size = len(Y)

npl = np.array([2, 1], dtype=ctypes.c_uint32)
raw_npl = np.ctypeslib.as_ctypes(npl)
mlp_model = my_lib.init_mlp(raw_npl, len(npl))

my_lib.train_mlp(mlp_model, X_flat_ptr, Y_flat_ptr, 3, 0.001, 100000, ctypes.c_bool(True))

background_points = []
background_colors = []
x1 = 0

while x1 < 3:
    x2 = 0
    while x2 < 3:
        background_points.append([x1, x2])
        x2 += 1.5
    x1 += 1.5

print(len(background_points))
for point in background_points:
    print(point)
    points_pointer = np.ctypeslib.as_ctypes(np.array(point, dtype=ctypes.c_float))
    prediction = my_lib.predict_mlp(mlp_model, points_pointer, 2, True)
    if prediction[0] > 0:
        background_colors.append('lightblue')
    else:
        background_colors.append('pink')

print(background_colors)

background_points = np.array(background_points)
plt.scatter(background_points[:, 0], background_points[:, 1], c=background_colors)
plt.scatter(X[0, 0], X[0, 1], color='blue')
plt.scatter(X[1:3, 0], X[1:3, 1], color='red')

plt.plot()
plt.show()
plt.clf()

my_lib.free_mlp(mlp_model)
