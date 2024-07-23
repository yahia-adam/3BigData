#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, save_rbf_model};
use std::ffi::CString;
use mylib::loads_serialized_ml_dataset;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 1000;
const CLUSTER_NUM: u32 = 100;
const GAMMA: f32 = 0.1;

const OUTPUT_DIR: &str = "../serialized_datasets";

fn load_model_dataset(model: &str) -> std::io::Result<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
    let file_path = format!("{}/{}.bin", OUTPUT_DIR, model);
    println!("Loading dataset: {}", file_path);
    loads_serialized_ml_dataset(&file_path)
}

fn test_model(model: *mut RadialBasisFunctionNetwork, x_test: &*const f32, y_test: &*const f32, input_count: usize) -> f32 {
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

    println!("Starting the training process...");

    println!("Loading metal_vs_other dataset...");
    let (metal_vs_other_train_images, metal_vs_other_train_labels, metal_vs_other_test_images, metal_vs_other_test_labels) = match load_model_dataset("metal_vs_other") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Serialization error for metal_vs_other: {}", e);
            return;
        }
    };

    println!("metal_vs_other dataset loaded successfully.");
    let metal_vs_other_train_data_size = metal_vs_other_train_labels.len();
    let metal_vs_other_test_data_size = metal_vs_other_test_labels.len();
    let metal_vs_other_input_count = metal_vs_other_train_images[0].len();
    let metal_vs_other_train_images_flaten: Vec<f32> = metal_vs_other_train_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let metal_vs_other_x_train_ptr: *const f32 = Vec::leak(metal_vs_other_train_images_flaten.clone()).as_ptr();
    let metal_vs_other_y_train_ptr: *const f32 = Vec::leak(metal_vs_other_train_labels.clone()).as_ptr();
    let metal_vs_other_test_images_flaten: Vec<f32> = metal_vs_other_test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let metal_vs_other_x_test_ptr: *const f32 = Vec::leak(metal_vs_other_test_images_flaten.clone()).as_ptr();
    let metal_vs_other_y_test_ptr: *const f32 = Vec::leak(metal_vs_other_test_labels.clone()).as_ptr();

    println!("Loading paper_vs_other dataset...");
    let (paper_vs_other_train_images, paper_vs_other_train_labels, paper_vs_other_test_images, paper_vs_other_test_labels) = match load_model_dataset("paper_vs_other") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Serialization error for paper_vs_other: {}", e);
            return;
        }
    };

    println!("paper_vs_other dataset loaded successfully.");
    let paper_vs_other_train_data_size = paper_vs_other_train_labels.len();
    let paper_vs_other_test_data_size = paper_vs_other_test_labels.len();
    let paper_vs_other_input_count = paper_vs_other_train_images[0].len();
    let paper_vs_other_train_images_flaten: Vec<f32> = paper_vs_other_train_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let paper_vs_other_x_train_ptr: *const f32 = Vec::leak(paper_vs_other_train_images_flaten.clone()).as_ptr();
    let paper_vs_other_y_train_ptr: *const f32 = Vec::leak(paper_vs_other_train_labels.clone()).as_ptr();
    let paper_vs_other_test_images_flaten: Vec<f32> = paper_vs_other_test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let paper_vs_other_x_test_ptr: *const f32 = Vec::leak(paper_vs_other_test_images_flaten.clone()).as_ptr();
    let paper_vs_other_y_test_ptr: *const f32 = Vec::leak(paper_vs_other_test_labels.clone()).as_ptr();

    println!("Loading plastic_vs_other dataset...");
    let (plastic_vs_other_train_images, plastic_vs_other_train_labels, plastic_vs_other_test_images, plastic_vs_other_test_labels) = match load_model_dataset("plastic_vs_other") {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Serialization error for plastic_vs_other: {}", e);
            return;
        }
    };

    println!("plastic_vs_other dataset loaded successfully.");
    let plastic_vs_other_train_data_size = plastic_vs_other_train_labels.len();
    let plastic_vs_other_test_data_size = plastic_vs_other_test_labels.len();
    let plastic_vs_other_input_count = plastic_vs_other_train_images[0].len();
    let plastic_vs_other_train_images_flaten: Vec<f32> = plastic_vs_other_train_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let plastic_vs_other_x_train_ptr: *const f32 = Vec::leak(plastic_vs_other_train_images_flaten.clone()).as_ptr();
    let plastic_vs_other_y_train_ptr: *const f32 = Vec::leak(plastic_vs_other_train_labels.clone()).as_ptr();
    let plastic_vs_other_test_images_flaten: Vec<f32> = plastic_vs_other_test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let plastic_vs_other_x_test_ptr: *const f32 = Vec::leak(plastic_vs_other_test_images_flaten.clone()).as_ptr();
    let plastic_vs_other_y_test_ptr: *const f32 = Vec::leak(plastic_vs_other_test_labels.clone()).as_ptr();

    let c_model_filename = CString::new(format!("../models/rbf/paper_vs_others_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(paper_vs_other_input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        paper_vs_other_x_train_ptr as *mut f32,
        paper_vs_other_y_train_ptr as *mut f32,
        EPOCHS as i32,
        LEARNING_RATE,
        paper_vs_other_input_count as i32,
        paper_vs_other_train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &paper_vs_other_x_test_ptr, &paper_vs_other_y_test_ptr, paper_vs_other_test_data_size);
    println!("Paper vs Others model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);

    let c_model_filename = CString::new(format!("../models/rbf/metal_vs_others_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(metal_vs_other_input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        metal_vs_other_x_train_ptr as *mut f32,
        metal_vs_other_y_train_ptr as *mut f32,
        EPOCHS as i32,
        LEARNING_RATE,
        metal_vs_other_input_count as i32,
        metal_vs_other_train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &metal_vs_other_x_test_ptr, &metal_vs_other_y_test_ptr, metal_vs_other_test_data_size);
    println!("Metal vs Others model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);

    let c_model_filename = CString::new(format!("../models/rbf/plastic_vs_others_lr={}_epochs={}.json", LEARNING_RATE, EPOCHS)).expect("CString::new failed");

    let model: *mut RadialBasisFunctionNetwork = init_rbf(plastic_vs_other_input_count as i32, CLUSTER_NUM as i32, GAMMA);

    train_rbf_rosenblatt(
        model,
        plastic_vs_other_x_train_ptr as *mut f32,
        plastic_vs_other_y_train_ptr as *mut f32,
        EPOCHS as i32,
        LEARNING_RATE,
        plastic_vs_other_input_count as i32,
        plastic_vs_other_train_data_size as i32,
    );

    save_rbf_model(model, c_model_filename.as_ptr());

    let accuracy = test_model(model, &plastic_vs_other_x_test_ptr, &plastic_vs_other_y_test_ptr, plastic_vs_other_test_data_size);
    println!("Plastic vs Others model - Training finished, Test accuracy: {:.2}%", accuracy * 100.0);

    free_rbf(model);

}
