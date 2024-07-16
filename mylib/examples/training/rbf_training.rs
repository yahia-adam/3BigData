use std::ffi::CString;
use std::path::PathBuf;

use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, save_rbf_model };

use mylib::{image_resize_vec, load_dataset};

fn main() {
    let base_dir = PathBuf::from("../mini_dataset");
    let train_path = base_dir.join("train");
    // let test_path = base_dir.join("test");

    let (train_images, train_labels)  = load_dataset(
        train_path.to_str().unwrap(),
        -1f32,
        -1f32,
        1f32
    );

    // let (test_images, test_labels) = load_dataset(
    //     test_path.to_str().unwrap(),
    //     -1f32,
    //     -1f32,
    //     1f32
    // );

    let train_data_size = train_labels.len();
    // let test_data_size = test_labels.len();
    let input_count = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *const f32 = Vec::leak(train_images_flatten).as_ptr();
    let y_train_ptr: *const f32 = Vec::leak(train_labels).as_ptr();

    // let test_images_flatten: Vec<f32> = test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    // let x_test_ptr: *const f32 = Vec::leak(test_images_flatten.clone()).as_ptr();
    // let y_test_ptr: *const f32 = Vec::leak(test_labels.clone()).as_ptr();

    let rbf_model: *mut RadialBasisFunctionNetwork = init_rbf(input_count as i32, 5, 1.5);
    train_rbf_rosenblatt(
        rbf_model,
        x_train_ptr as *mut f32,
        y_train_ptr as *mut f32,
        10000000,
        0.1,
        input_count as i32,
        train_data_size as i32
    );

    let model_path = "../models/rbf_train.json";
    let model_path_cstr = CString::new(model_path).expect("CString::new failed");
    save_rbf_model(
        rbf_model,
        model_path_cstr.as_ptr(),
    );

    println!("metal");
    let metal1 = "../dataset/train/metal/metal_1025.jpg";
    let test1 = predict_rbf_classification(rbf_model,image_resize_vec(metal1, 32).as_mut_ptr());
    println!("metal - test1 = {}", test1);

    println!("paper");
    let paper1 = "../dataset/train/paper/paper_3044.jpg";
    let test2 = predict_rbf_classification(rbf_model,image_resize_vec(paper1, 32).as_mut_ptr());
    println!("paper - test2 = {}", test2);

    println!("plastic");
    let plastic1 = "../dataset/train/plastic/plastic_6064.jpg";
    let test3 = predict_rbf_classification(rbf_model,image_resize_vec(plastic1, 32).as_mut_ptr());
    println!("plastic - test3 = {}", test3);

    free_rbf(rbf_model);
}

