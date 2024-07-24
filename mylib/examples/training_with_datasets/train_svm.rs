use std::ffi::CString;
#[allow(unused_imports)]

use mylib::{init_svm, SVMModel, train_svm, loads_serialized_ml_dataset,};
use mylib::support_vector_machine::save_svm;

fn load_model_dataset(model: &str) -> std::io::Result<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
    let file_path = format!("{}/{}.bin", OUTPUT_DIR, model);
    println!("Loading dataset: {}", file_path);
    loads_serialized_ml_dataset(&file_path)
}

const LEARNING_RATES: &[f32] = &[10e-8];
const EPOCHS: &[u32] = &[1000];

const OUTPUT_DIR: &str = "../serialized_datasets";

const KERNEL:u32 = 3;
const KERNEL_VALUE:f32 = 30.0;
const C:f32 = 0.5;
const EPSILON:f32 =1e-4;

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

    //let total_iterations = LEARNING_RATES.len() * EPOCHS.len() * 3;
    //let mut current_iteration = 0;

    //for lr in LEARNING_RATES {
      //  for epoch in EPOCHS {
//            current_iteration += 1;
  //          println!("\nIteration {}/{}", current_iteration, total_iterations);

            //let c_log_filename =  CString::new(format!("../logs/ml/{}:lr={}epochs={}", "metal_vs_other", lr, epoch)).expect("CString::new failed");
            let c_model_filename = CString::new(format!("../models/svm/{}=kernel={}kernel_value={}c={}epsilon={}.json", "metal_vs_other",KERNEL, KERNEL_VALUE, C, EPSILON)).expect("CString::new failed");
            let model: *mut SVMModel = init_svm(metal_vs_other_input_count as u32, KERNEL, KERNEL_VALUE);
            train_svm(model, metal_vs_other_x_train_ptr, metal_vs_other_y_train_ptr, metal_vs_other_train_data_size as u32, C, EPSILON, metal_vs_other_x_test_ptr, metal_vs_other_y_test_ptr, metal_vs_other_test_data_size as u32);
            save_svm(model, c_model_filename.as_ptr());
            println!("metal_vs_other model trained and saved.");

//            current_iteration += 1;
//            println!("\nIteration {}/{}", current_iteration, total_iterations);

            println!("Training paper_vs_other model");
            //let c_log_filename =  CString::new(format!("../logs/ml/{}:lr={}epochs={}", "paper_vs_other", lr, epoch)).expect("CString::new failed");
            let c_model_filename = CString::new(format!("../models/svm/{}=kernel={}kernel_value={}c={}epsilon={}.json", "paper_vs_other",KERNEL, KERNEL_VALUE, C, EPSILON)).expect("CString::new failed");
            let model: *mut SVMModel = init_svm(paper_vs_other_input_count as u32, KERNEL, KERNEL_VALUE);
            train_svm(model, paper_vs_other_x_train_ptr, paper_vs_other_y_train_ptr, paper_vs_other_train_data_size as u32, C, EPSILON, metal_vs_other_x_test_ptr, metal_vs_other_y_test_ptr, metal_vs_other_test_data_size as u32);

            save_svm(model, c_model_filename.as_ptr());
            println!("paper_vs_other model trained and saved.");

            println!("Training plastic_vs_other model");
            let c_model_filename = CString::new(format!("../models/svm/{}=kernel={}kernel_value={}c={}epsilon={}.json", "plastic_vs_other",KERNEL, KERNEL_VALUE, C, EPSILON)).expect("CString::new failed");
            let model: *mut SVMModel = init_svm(plastic_vs_other_input_count as u32, KERNEL, KERNEL_VALUE);
            train_svm(model, plastic_vs_other_x_train_ptr, plastic_vs_other_y_train_ptr, plastic_vs_other_train_data_size as u32, C, EPSILON, metal_vs_other_x_test_ptr, metal_vs_other_y_test_ptr, metal_vs_other_test_data_size as u32);

            save_svm(model, c_model_filename.as_ptr());
            println!("plastic_vs_other model trained and saved.");
   //     }
 //   }

    println!("All models have been trained and saved successfully.");
}