use std::ffi::CString;
use std::path::PathBuf;
#[allow(unused_imports)]
use mylib::{
    LinearModel,
    init_linear_model,
    save_linear_model,
    train_linear_model
};
use mylib::load_dataset;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100_000;
fn main() {

    let base_dir = PathBuf::from("../dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    let (train_images, train_labels)  = load_dataset(
        train_path.to_str().unwrap(),
        -1f32,
        1f32,
        -1f32
    );
    let (test_images, test_labels) = load_dataset(
        test_path.to_str().unwrap(),
        -1f32,
        1f32,
        -1f32
    );

    let train_data_size = train_labels.len();
    let test_data_size = test_labels.len();
    let input_count = train_images[0].len();

    let train_images_flaten: Vec<f32> = train_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_train_ptr: *const f32 = Vec::leak(train_images_flaten.clone()).as_ptr();
    let y_train_ptr: *const f32 = Vec::leak(train_labels.clone()).as_ptr();

    let test_images_flaten: Vec<f32> = test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_test_ptr: *const f32 = Vec::leak(test_images_flaten.clone()).as_ptr();
    let y_test_ptr: *const f32 = Vec::leak(test_labels.clone()).as_ptr();

    let c_log_filename =  CString::new(format!("../logs/ml/metale_vs_other:lr={}epochs={}", LEARNING_RATE, EPOCHS)).expect("CString::new failed");
    let c_model_filename = CString::new(format!("../models/ml/classification/metale_vs_other:lr={}epochs{}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let metal_vs_other: *mut LinearModel = init_linear_model(input_count as u32, true, false);
    train_linear_model(
        metal_vs_other,
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
        true,
        true,
        true,
    );

    println!("model1_finish");
}
