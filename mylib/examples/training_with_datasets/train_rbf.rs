#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, save_rbf_model, load_ml_dataset};
use std::ffi::CString;
use std::path::PathBuf;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
const CLUSTER_NUM: u32 = 10;
const GAMMA: f32 = 0.1;

fn test_model(model: *mut RadialBasisFunctionNetwork, x_test: &[f32], y_test: &[f32], input_count: usize) -> f32 {
    let mut correct_predictions = 0;
    let total_predictions = y_test.len();

    for i in 0..total_predictions {
        let start = i * input_count;
        let end = start + input_count;
        let input = &x_test[start..end];
        let prediction = predict_rbf_classification(model, input.as_ptr() as *mut f32);
        let true_label = y_test[i];

        if (prediction > 0.0 && true_label > 0.0) || (prediction <= 0.0 && true_label <= 0.0) {
            correct_predictions += 1;
        }
    }

    (correct_predictions as f32) / (total_predictions as f32)
}

fn main() {
    let base_dir = PathBuf::from("../dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    // train and test paper vs metal

    let (train_images, train_labels) = load_ml_dataset(
        train_path.to_str().unwrap(),
        1.0,
        -1.0,
        -1.0
    );
    let (test_images, test_labels) = load_ml_dataset(
        test_path.to_str().unwrap(),
        1.0,
        -1.0,
        -1.0
    );

    let train_data_size = train_labels.len();
    let input_count = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *mut f32 = Vec::leak(train_images_flatten).as_mut_ptr();
    let y_train_ptr: *mut f32 = Vec::leak(train_labels).as_mut_ptr();

    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_labels_flatten: Vec<f32> = test_labels;

    let c_model_filename = CString::new(format!("../models/rbf/paper_vs_metal_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        x_train_ptr,
        y_train_ptr,
        EPOCHS as i32,
        LEARNING_RATE,
        input_count as i32,
        train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &test_images_flatten, &test_labels_flatten, input_count);
    println!("Paper vs Metal model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);


    // train and test plastic vs paper

    let (train_images, train_labels) = load_ml_dataset(
        train_path.to_str().unwrap(),
        -1.0,
        1.0,
        -1.0
    );
    let (test_images, test_labels) = load_ml_dataset(
        test_path.to_str().unwrap(),
        -1.0,
        1.0,
        -1.0
    );

    let train_data_size = train_labels.len();
    let input_count = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *mut f32 = Vec::leak(train_images_flatten).as_mut_ptr();
    let y_train_ptr: *mut f32 = Vec::leak(train_labels).as_mut_ptr();

    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_labels_flatten: Vec<f32> = test_labels;

    let c_model_filename = CString::new(format!("../models/rbf/plastic_vs_paper_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        x_train_ptr,
        y_train_ptr,
        EPOCHS as i32,
        LEARNING_RATE,
        input_count as i32,
        train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &test_images_flatten, &test_labels_flatten, input_count);
    println!("Plastic vs Paper model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);


    // train and test plastic vs metal

    let (train_images, train_labels) = load_ml_dataset(
        train_path.to_str().unwrap(),
        -1.0, // plastic
        1.0,  // metal
        -1.0
    );
    let (test_images, test_labels) = load_ml_dataset(
        test_path.to_str().unwrap(),
        -1.0,
        1.0,
        -1.0
    );

    let train_data_size = train_labels.len();
    let input_count = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *mut f32 = Vec::leak(train_images_flatten).as_mut_ptr();
    let y_train_ptr: *mut f32 = Vec::leak(train_labels).as_mut_ptr();

    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let test_labels_flatten: Vec<f32> = test_labels;

    let c_model_filename = CString::new(format!("../models/rbf/plastic_vs_metal_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        x_train_ptr,
        y_train_ptr,
        EPOCHS as i32,
        LEARNING_RATE,
        input_count as i32,
        train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &test_images_flatten, &test_labels_flatten, input_count);
    println!("Plastic vs Metal model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);

}
