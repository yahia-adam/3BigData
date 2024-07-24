import ctypes
import sys


def init_lib():
    if sys.platform == 'win32':
        lib_path = r"../mylib/target/release/mylib.dll"
    else:
        lib_path = "../mylib/target/release/libmylib.so"

    my_lib = ctypes.cdll.LoadLibrary(lib_path)

    # -------------------------- lm --------------------------
    """ 
    init_linear_model(
        input_count: u32,
        is_classification: bool,
        is_multiclass: bool
    ) -> *mut LinearModel
    """

    my_lib.init_linear_model.argtypes = [
        ctypes.c_uint32,
        ctypes.c_bool,
        ctypes.c_bool
    ]

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
        log_filename: *const c_char,
        model_filename: *const c_char,
        display_loss: bool,
        display_tensorboad: bool,
        save_model: bool,
        )
    """

    my_lib.train_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
    ]

    my_lib.train_linear_model.restype = None

    """
    predict_linear_model(
        model: *mut LinearModel,
        inputs: *mut f32
    ) -> c_float 
    """

    my_lib.predict_linear_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float)
    ]

    my_lib.predict_linear_model.restype = ctypes.c_float

    """
    free_linear_model(
        model: *mut LinearModel
    )
    """

    my_lib.free_linear_model.argtypes = [
        ctypes.c_void_p
    ]

    my_lib.free_linear_model.restype = None

    """
    loads_linear_model(
        json_str_ptr: *const c_char
    ) -> *mut LinearModel
    """

    my_lib.loads_linear_model.argtypes = [
        ctypes.POINTER(ctypes.c_char)
    ]

    my_lib.loads_linear_model.restype = ctypes.c_void_p

    # -------------------------- mlp --------------------------
    """
    init_mlp(
        npl: *const u32,
        npl_size: u32,
        is_classification: bool,
    ) -> *mut MultiLayerPerceptron
    """

    my_lib.init_mlp.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
        ctypes.c_bool
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
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
    ]

    my_lib.train_mlp.restype = ctypes.c_bool

    """
    predict_mlp(
        model: *mut MultiLayerPerceptron,
        sample_inputs: *const f32,
    ) -> *mut f32 {
    """

    my_lib.predict_mlp.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float)
    ]

    my_lib.predict_mlp.restype = ctypes.POINTER(ctypes.c_float)

    """
    free_mlp(
        ptr: *mut MultiLayerPerceptron
    )
    """

    my_lib.free_mlp.argtypes = [
        ctypes.c_void_p
    ]
    
    my_lib.free_mlp.restype = None

    """
    loads_mlp_model(
        filepath: *const c_char
    ) -> *mut MultiLayerPerceptron
    """

    my_lib.loads_mlp_model.argtypes = [
        ctypes.POINTER(ctypes.c_char)
    ]

    my_lib.loads_mlp_model.restype = ctypes.c_void_p

    # ---------------------------- rbf --------------------------
    """
    init_rbf(
        input_dim : i32, 
        cluster_num : i32, 
        gamma : f32
    ) -> *mut RadialBasisFunctionNetwork
    """

    my_lib.init_rbf.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float
    ]

    my_lib.init_rbf.restype = ctypes.c_void_p

    """
    train_rbf_regression(
        model : *mut RadialBasisFunctionNetwork, 
        sample_inputs_flat : *mut f32,
        expected_outputs : *mut f32, 
        inputs_size : i32, 
        sample_count : i32
    )
    """

    my_lib.train_rbf_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32
    ]

    my_lib.train_rbf_regression.restype = None

    """
    predict_rbf_regression(
        model : *mut RadialBasisFunctionNetwork, 
        inputs : *mut f32
    ) -> f32
    """

    my_lib.predict_rbf_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float)
    ]

    my_lib.predict_rbf_regression.restype = ctypes.c_float

    """
    train_rbf_rosenblatt(
        model : *mut RadialBasisFunctionNetwork, 
        sample_inputs_flat : *mut f32,
        expected_outputs : *mut f32, 
        iterations_count : i32, 
        alpha : f32, 
        inputs_size : i32, 
        sample_count : i32
    )
    """

    my_lib.train_rbf_rosenblatt.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.c_int32
    ]

    my_lib.train_rbf_rosenblatt.restype = None

    """
    predict_rbf_classification(
        model : *mut RadialBasisFunctionNetwork, 
        inputs : *mut f32
    )-> f32
    """

    my_lib.predict_rbf_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float)
    ]

    my_lib.predict_rbf_classification.restype = ctypes.c_float

    """
    free_rbf(
        model : *mut RadialBasisFunctionNetwork
    )
    """

    my_lib.free_rbf.argtypes = [ctypes.c_void_p]

    my_lib.free_rbf.restype = None
    
    """
    load_rbf_model(
        filepath: *const c_char
    ) -> *mut RadialBasisFunctionNetwork
    """

    my_lib.load_rbf_model.argtypes = [
        ctypes.POINTER(ctypes.c_char)
    ]

    my_lib.load_rbf_model.restype = ctypes.c_void_p

    return my_lib
