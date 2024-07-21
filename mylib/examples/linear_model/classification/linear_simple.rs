use std::ffi::CString;
#[allow(unused_imports)]
use mylib::{
    init_linear_model, load_linear_model, predict_linear_model, save_linear_model,
    train_linear_model, LinearModel,
};
const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0]
    ];
    let y: Vec<f32> = vec![
        1.0,
        -1.0,
        -1.0
    ];
    let train_data_size = x.len();
    let test_data_size = x.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_train_ptr: *const f32 = Vec::leak(x_flaten.clone()).as_ptr();
    let y_train_ptr: *const f32 = Vec::leak(y.clone()).as_ptr();

    let x_test_ptr: *const f32 = Vec::leak(x_flaten.clone()).as_ptr();
    let y_test_ptr: *const f32 = Vec::leak(y.clone()).as_ptr();

    let linear_model: *mut LinearModel = init_linear_model(2, true, false);

    let c_log_filename =  CString::new(format!("../logs/ml/dim=2lr={}epochs={}", LEARNING_RATE, EPOCHS)).expect("CString::new failed");
    let c_model_filename = CString::new(format!("../models/examples/mlp/classification/dim=2lr={}epochs{}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    train_linear_model(
        linear_model,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        LEARNING_RATE,
        EPOCHS,
        c_log_filename.as_ptr(),
        c_model_filename.as_ptr(),
        false,
        false,
        true,
    );
}
