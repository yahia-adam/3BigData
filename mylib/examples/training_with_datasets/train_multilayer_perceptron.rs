use std::ffi::CString;
use std::path::PathBuf;
#[allow(unused_imports)]
use mylib::{
    MultiLayerPerceptron,
    init_mlp,
    train_mlp,
    predict_mlp,
    free_mlp,
    save_mlp_model,
};
use mylib::{image_resize_vec, load_mlp_dataset};

const LEARNING_RATE: f32 = 0.0001;
const EPOCHS: u32 = 500;
const DIM:  &[u32] = &[128, 64, 32, 3];

fn main() {

    println!("{:?}",format!("dim={:?}epoch={}lr={}", DIM,EPOCHS,LEARNING_RATE));
    let base_dir = PathBuf::from("../dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");
    
    println!("Loading train dataset...");
    let (train_images, train_labels) = load_mlp_dataset(
        train_path.to_str().unwrap(),
    );

    println!("Loading test dataset...");
    let (test_images, test_labels) = load_mlp_dataset(
        test_path.to_str().unwrap(),
    );
    
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
    
    let mut npl: Vec<u32> = vec![input_size as u32];
    npl.extend(DIM);

    let npl_size = npl.len();
    let model: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl_size as u32, true);
    
    println!("Finished initializing model");

    let model_prameter: String = format!("dim={:?}epoch={}lr={}", DIM,EPOCHS,LEARNING_RATE);
    let c_log_filename: CString = CString::new(model_prameter.clone()).expect("CString::new failed");
    let c_model_filename: CString = CString::new(format!("../models/dataset/mlp/{}.json", model_prameter) ).expect("CString::new failed");

    unsafe {
        let success = train_mlp(
            model,
            x_train_ptr,
            y_train_ptr,
            train_data_size as u32,
            x_test_ptr,
            y_test_ptr,
            test_data_size as u32,
            LEARNING_RATE,
            EPOCHS,
            c_log_filename.as_ptr(),
            c_model_filename.clone().as_ptr(),
            true,
            true,
            true,
        );
        
        if success {
            println!("Training completed successfully");

            save_mlp_model(model, c_model_filename.as_ptr());
            
            println!("Testing on a metal image:");
            let metal1 = "../dataset/train/metal/metal_1025.jpg";
            let metal_image = image_resize_vec(metal1, 32);
            let res = predict_mlp(model, metal_image.as_ptr());
            if !res.is_null() {
                let res: Vec<f32> = Vec::from_raw_parts(res, 3, 3);
                println!("paper: -1\nmetal: 0\nplastic: 1\nPrediction: {:?}", res);
            }
        } else {
            println!("Training failed");
        }
        
        free_mlp(model);
    }
}