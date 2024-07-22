#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, save_rbf_model, load_mlp_dataset};
use std::ffi::CString;
use std::path::PathBuf;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
const CLUSTER_NUM: u32 = 100;

const GAMMA: f32 = 1.0;

fn preprocess_image(path: &str) -> Vec<f32> {

    let img = image::open(path).expect("Failed to open image");

    let img_gray = img.to_luma8();
    let img_resized = image::imageops::resize(&img_gray, 32, 32, image::imageops::FilterType::Lanczos3);

    let flat_img: Vec<f32> = img_resized.into_vec().iter().map(|&p| p as f32 / 255.0).collect();

    flat_img
}

fn interpret_prediction(pred: f32) -> &'static str {
    if pred > 0.0 {
        "metal"
    } else if pred < -0.5 {
        "plastic"
    } else {
        "paper"
    }
}

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
    //let test_data_size = test_labels.len();
    let input_size = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *const f32 = train_images_flatten.as_ptr();
    let train_labels_flatten: Vec<f32> = train_labels.into_iter().flatten().collect();
    let y_train_ptr: *const f32 = train_labels_flatten.as_ptr();

    //let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    //let test_label_flatten: Vec<f32> = test_labels.into_iter().flatten().collect();

    let model: *mut RadialBasisFunctionNetwork = init_rbf(input_size as i32, CLUSTER_NUM as i32, GAMMA);

    println!("Finished initializing model");

    let model_parameter: String = format!("Variation_couches:dim={:?}epoch={}lr={}", GAMMA, EPOCHS, LEARNING_RATE);
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

    println!("metal");
    let metal_paths = vec!["../dataset/test/metal/metal_10.png",
                           "../dataset/test/metal/metal_1281.jpg",
                           "../dataset/test/metal/metal_1444.jpg"];

    for path in metal_paths {
        let preprocessed_image = preprocess_image(path);
        let test = predict_rbf_classification(model, preprocessed_image.as_ptr() as *mut f32);
        let class = interpret_prediction(test);
        println!("{} - prediction = {}", path, class);
    }

    println!("paper");
    let paper_paths = vec!["../dataset/test/paper/paper_10.jpg",
                           "../dataset/test/paper/paper_1087.jpg",
                           "../dataset/test/paper/paper_1305.jpg"];

    for path in paper_paths {
        let preprocessed_image = preprocess_image(path);
        let test = predict_rbf_classification(model, preprocessed_image.as_ptr() as *mut f32);
        let class = interpret_prediction(test);
        println!("{} - prediction = {}", path, class);
    }

    println!("plastic");
    let plastic_paths = vec!["../dataset/test/plastic/plastic_10.jpg",
                             "../dataset/test/plastic/plastic_1354.jpg",
                             "../dataset/test/plastic/plastic_1638.jpg"];

    for path in plastic_paths {
        let preprocessed_image = preprocess_image(path);
        let test = predict_rbf_classification(model, preprocessed_image.as_ptr() as *mut f32);
        let class = interpret_prediction(test);
        println!("{} - prediction = {}", path, class);
    }

    free_rbf(model);
}

