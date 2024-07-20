import ctypes
import sys


def init_lib():
    if sys.platform == 'win32':
        lib_path = "../../mylib/target/release/mylib.dll"
    else:
        lib_path = "../../mylib/target/release/libmylib.so"

    my_lib = ctypes.cdll.LoadLibrary(lib_path)

    """ 
    init_linear_model(
        input_count: u32,
        is_classification: bool,
        is_multiclass: bool
    ) -> *mut LinearModel
    """

    my_lib.init_linear_model.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_bool]
    my_lib.init_linear_model.restype = ctypes.c_void_p

    """ 
    train_linear_model( 
        model: *mut LinearModel,
        x_train: *const c_float,
        y_train: *const c_float,
        train_data_size: u32,
        x_test: *const c_float,
        y_test: *const c_float,
        test_data_size: u32,
        learning_rate: f32,
        epochs: u32,
        log_filename: *const c_char)
    """

    my_lib.train_linear_model.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_float),  # x_train
        ctypes.POINTER(ctypes.c_float),  # y_train
        ctypes.c_uint32,  # train_data_size
        ctypes.POINTER(ctypes.c_float),  # x_test
        ctypes.POINTER(ctypes.c_float),  # y_test
        ctypes.c_uint32,  # test_data_size
        ctypes.c_float,  # learning_rate
        ctypes.c_uint32,  # epochs
        ctypes.POINTER(ctypes.c_char)  # log_filename
    ]

    my_lib.train_linear_model.restype = None

    """
    predict_linear_model(
        model: *mut LinearModel,
        inputs: *mut f32
    ) -> c_float 
    """

    my_lib.predict_linear_model.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_float)  # inputs
    ]

    my_lib.predict_linear_model.restype = ctypes.c_float

    """
    free_linear_model(
        model: *mut LinearModel
    )
    """

    my_lib.free_linear_model.argtypes = [
        ctypes.c_void_p  # model
    ]

    my_lib.free_linear_model.restype = None

    # pub extern "C" fn load_linear_model(json_str_ptr: *const c_char) -> *mut LinearModel
    my_lib.load_linear_model.argtypes = [ctypes.POINTER(ctypes.c_char)]
    my_lib.load_linear_model.restype = ctypes.c_void_p

    # -------------------------- init mlp --------------------------
    """
    init_mlp(
        npl: *const u32,
        npl_size: u32,
        is_classification: bool,
    ) -> *mut MultiLayerPerceptron
    """

    my_lib.init_mlp.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # model
        ctypes.c_uint32,  # npl_size
        ctypes.c_bool  # is classification
    ]

    my_lib.init_mlp.restype = ctypes.c_void_p

    """
    train_mlp(
        model: *mut MultiLayerPerceptron,
        x_train: *const c_float,
        y_train: *const c_float,
        train_data_size: u32,
        x_test: *const c_float,
        y_test: *const c_float,
        test_data_size: u32,
        learning_rate: f32,
        epochs: u32,
        log_filename: *const c_char,
        model_filename: *const c_char,
        display_loss: bool,
        display_tensorboad: bool,
        save_model: bool,
    ) -> bool
    """

    my_lib.train_mlp.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_float),  # x_train
        ctypes.POINTER(ctypes.c_float),  # y_train
        ctypes.c_uint32,  # train_data_size
        ctypes.POINTER(ctypes.c_float),  # x_test
        ctypes.POINTER(ctypes.c_float),  # y_test
        ctypes.c_uint32,  # test_data_size
        ctypes.c_float,  # learning_rate
        ctypes.c_uint32,  # epochs
        ctypes.c_char_p,  # log_filename
        ctypes.c_char_p,  # model_filename
        ctypes.c_bool,  # display_loss
        ctypes.c_bool,  # display_tensorboad
        ctypes.c_bool,  # save_model
    ]
    my_lib.train_mlp.restype = ctypes.c_bool

    """
    predict_mlp(
        model: *mut MultiLayerPerceptron,
        sample_inputs: *const f32,
    ) -> *mut f32 {
    """

    my_lib.predict_mlp.argtypes = [
        ctypes.c_void_p,  # model
        ctypes.POINTER(ctypes.c_float)  # inputs
    ]

    my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

    """
    free_mlp(
        ptr: *mut MultiLayerPerceptron
    )
    """

    my_lib.free_mlp.argtypes = [
        ctypes.c_void_p  # model
    ]
    
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
