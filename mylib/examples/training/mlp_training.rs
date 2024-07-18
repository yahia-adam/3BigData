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
use mylib::{image_resize_vec, load_dataset};

fn main() {
    let base_dir = PathBuf::from("../mini_dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");
    
    println!("Loading train dataset...");
    let (train_images, train_labels) = load_dataset(
        train_path.to_str().unwrap(),
        -1.0,
        0.0,
        1.0
    );
    
    println!("Loading test dataset...");
    let (test_images, test_labels) = load_dataset(
        test_path.to_str().unwrap(),
        -1.0,
        0.0,
        1.0
    );
    
    println!("Finished loading dataset");
    
    let train_data_size = train_labels.len();
    let test_data_size = test_labels.len();
    let input_size = train_images[0].len();
    
    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *const f32 = train_images_flatten.as_ptr();
    let y_train_ptr: *const f32 = train_labels.as_ptr();
    
    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let x_test_ptr: *const f32 = test_images_flatten.as_ptr();
    let y_test_ptr: *const f32 = test_labels.as_ptr();
    
    let npl: Vec<u32> = vec![input_size as u32, 64, 32, 3];
    let npl_size = npl.len();
    let model: *mut MultiLayerPerceptron = unsafe { init_mlp(npl.as_ptr(), npl_size as u32, true) };
    
    println!("Finished initializing model");
    
    unsafe {
        let success = train_mlp(
            model,
            x_train_ptr,
            y_train_ptr,
            train_data_size as u32,
            x_test_ptr,
            y_test_ptr,
            test_data_size as u32,
            0.001,
            10_000
        );
        
        if success {
            println!("Training completed successfully");
            
            let model_path = "../models/pmc_dataset.json";
            let model_path_cstr = CString::new(model_path).expect("CString::new failed");
            save_mlp_model(model, model_path_cstr.as_ptr());
            
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