import ctypes
import sys


def init_lib():
    if sys.platform == 'win32':
        lib_path = "../../mylib/target/release/mylib.dll"
    else:
        lib_path = "../../mylib/target/release/libmylib.so"

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
    # pub extern "C" fn load_linear_model(json_str_ptr: *const c_char) -> *mut LinearModel
    my_lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    my_lib.load_linear_model.restype = ctypes.c_void_p
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
    # pub extern "C" fn loads_mlp_model(filepath: *const c_char) -> *mut MultiLayerPerceptron
    my_lib.load_mlp.argtypes = [ctypes.POINTER(ctypes.c_char)]
    my_lib.load_mlp.restype = ctypes.c_void_p
    # ---------------------------- init RBF --------------------------
    # pub extern "C" fn init_rbf(input_dim : i32, cluster_num : i32, gamma : f32) -> *mut RadicalBasisFunctionNetwork
    my_lib.init_rbf.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_float]
    my_lib.init_rbf.restype = ctypes.c_void_p
    # pub extern "C" fn train_rbf_regression(model : *mut RadicalBasisFunctionNetwork, sample_inputs_flat : *mut f32,
    #                                           expected_outputs : *mut f32, inputs_size : i32, sample_count : i32)
    my_lib.train_rbf_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int32, ctypes.c_int32]
    my_lib.train_rbf_regression.restype = None
    # pub extern "C" fn predict_rbf_regression(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32) -> f32
    my_lib.predict_rbf_regression.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    my_lib.predict_rbf_regression.restype = ctypes.c_float
    # pub extern "C" fn train_rbf_rosenblatt(model : *mut RadicalBasisFunctionNetwork, sample_inputs_flat : *mut f32,
    #           expected_outputs : *mut f32, iterations_count : i32, alpha : f32, inputs_size : i32, sample_count : i32)
    my_lib.train_rbf_rosenblatt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32]
    my_lib.train_rbf_rosenblatt.restype = None
    # pub extern "C" fn predict_rbf_classification(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32)-> f32
    my_lib.predict_rbf_classification.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    my_lib.predict_rbf_classification.restype = ctypes.c_float
    # pub extern "C" fn free_rbf(model : *mut RadicalBasisFunctionNetwork)
    my_lib.free_rbf.argtypes = [ctypes.c_void_p]
    my_lib.free_rbf.restype = None

    return my_lib
