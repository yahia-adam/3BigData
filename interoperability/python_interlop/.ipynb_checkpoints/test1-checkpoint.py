import matplotlib.pyplot as plt
import numpy as np
import ctypes

def init_lib():
    lib_path = "./../../mylib/target/debug/libmylib.so"
    my_lib = ctypes.cdll.LoadLibrary(lib_path)

    my_lib.hello.argtypes = []
    my_lib.hello.restype = None

    my_lib.init_mlp.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    my_lib.init_mlp.restype = ctypes.c_void_p

    my_lib.train_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_float, ctypes.c_uint32, ctypes.c_bool]
    my_lib.train_mlp.restype = None

    my_lib.predict_mlp.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
    my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

    my_lib.free_mlp.argtypes = [ctypes.c_void_p]
    my_lib.free_mlp.restype = None
    return my_lib

if __name__ == '__main__':
    my_lib = init_lib()
    # my_lib.hello()

    # # Initialize the MLP
    npl = np.array([2, 3, 1], dtype=ctypes.c_uint32)
    raw_npl = np.ctypeslib.as_ctypes(npl)
    model = my_lib.init_mlp(raw_npl, len(npl))

    # # Train the MLP
    X = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype=ctypes.c_float)
    Y = np.array([ 1.0, -1.0, -1.0], dtype=ctypes.c_float)
    X_p = np.ctypeslib.as_ctypes(X)
    Y_p = np.ctypeslib.as_ctypes(Y)
    my_lib.train_mlp(model, X_p, Y_p, 2, 1, 3, 0.01, 10000, ctypes.c_bool(True))

    # Test input for prediction
    test_input = np.array([2.0, 2.0], dtype=ctypes.c_float)
    test_input_p = np.ctypeslib.as_ctypes(test_input)
    predictions = my_lib.predict_mlp(model, test_input_p, 2, True)
    print("Prediction for [2.0, 2.0]:", predictions[0])  # Access the prediction result
    my_lib.free_mlp(model)