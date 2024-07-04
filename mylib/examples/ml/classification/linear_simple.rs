#[allow(unused_imports)]
use mylib::{
    init_linear_model, load_linear_model, predict_linear_model, save_linear_model,
    train_linear_model, LinearModel,
};

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

    let linear_model: *mut LinearModel = init_linear_model(2, true);
    train_linear_model(
        linear_model,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        0.001, 10_000);
}
