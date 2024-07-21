#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, save_rbf_model, load_mlp_dataset};
use std::ffi::CString;
use std::path::PathBuf;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
const CLUSTER_NUM: u32 = 100;


fn main(){
    let base_dir = PathBuf::from("../dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    println!("Loading train dataset...");
    let (train_images, train_labels) = load_mlp_dataset(train_path.to_str().unwrap());

    println!("Loading test dataset...");
    let (test_images, test_labels) = load_mlp_dataset(test_path.to_str().unwrap());

    println!("Finished loading dataset");

    let train_data_size = train_labels.len();
    let test_data_size = test_labels.len();
    let input_size = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *const f32 = train_images_flatten.as_ptr();
    let train_labels_flatten: Vec<f32> = train_labels.into_iter().flatten().collect();
    let y_train_ptr: *const f32 = train_labels_flatten.as_ptr();

    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let x_test_ptr: *const f32 = test_images_flatten.as_ptr();
    let test_label_flatten: Vec<f32> = test_labels.into_iter().flatten().collect();
    let y_test_ptr: *const f32 = test_label_flatten.as_ptr();


    let gamma = vec![
        0.001, 0.01, 0.1, 1.0
    ];

    for g in gamma {
        let model: *mut RadialBasisFunctionNetwork = init_rbf(input_size as i32, CLUSTER_NUM as i32, g);

        println!("Finished initializing model");

        let model_parameter: String = format!("Variation_couches:dim={:?}epoch={}lr={}", g, EPOCHS, LEARNING_RATE);
        let c_log_filename: CString =
            CString::new(model_parameter.clone()).expect("CString::new failed");
        let c_model_filename: CString =
            CString::new(format!("../models/dataset/rbf/{}.json", model_parameter))
                .expect("CString::new failed");

        train_rbf_rosenblatt(
            model,
            x_train_ptr as *mut f32,
            y_train_ptr as *mut f32,
            EPOCHS as i32,
            LEARNING_RATE,
            input_size as i32,
            train_data_size as i32,
        );

        save_rbf_model(model, c_model_filename.as_ptr());

        free_rbf(model);
    }

}