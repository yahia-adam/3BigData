use std::ffi::CString;
use std::path::PathBuf;
#[allow(unused_imports)]
use mylib::{
    LinearModel,
    init_linear_model,
    save_linear_model,
    train_linear_model
};
use mylib::{image_resize_vec, load_ml_dataset, predict_linear_model};

fn main() {

    let base_dir = PathBuf::from("../mini_dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    let (train_images, train_labels)  = load_ml_dataset(
        train_path.to_str().unwrap(),
        -1f32,
        -1f32,
        1f32
    );
    let (test_images, test_labels) = load_ml_dataset(
        test_path.to_str().unwrap(),
        -1f32,
        -1f32,
        1f32
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


    let log_filename = "plastic_vs_other";
    let log_filename = CString::new(log_filename).expect("CString::new failed");
    let plastic_vs_other: *mut LinearModel = init_linear_model(input_count as u32, true, false);
    train_linear_model(
        plastic_vs_other,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        0.001, 1_000_000,
        log_filename.as_ptr()
    );
    let model_path = "../models/mlp_plastic_vs_other.json";
    let model_path_cstr = CString::new(model_path).expect("CString::new failed");
    save_linear_model(
        plastic_vs_other,
        model_path_cstr.as_ptr()
    );


    println!("model1_finish");

    let (train_images, train_labels)  = load_ml_dataset(
        train_path.to_str().unwrap(),
        -1f32,
        1f32,
        -1f32
    );
    let (test_images, test_labels) = load_ml_dataset(
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

    let metal_vs_other: *mut LinearModel = init_linear_model(input_count as u32, true, false);

    let log_filename = "metal_vs_other";
    let log_filename = CString::new(log_filename).expect("CString::new failed");
    train_linear_model(
        metal_vs_other,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        0.001, 1_000_000,
        log_filename.as_ptr()
    );
    let model_path = "../models/mlp_metal_vs_other.json";
    let model_path_cstr = CString::new(model_path).expect("CString::new failed");
    save_linear_model(
        metal_vs_other,
        model_path_cstr.as_ptr()
    );

    println!("model2_finish");

    let (train_images, train_labels)  = load_ml_dataset(
        train_path.to_str().unwrap(),
        1f32,
        -1f32,
        -1f32
    );
    let (test_images, test_labels) = load_ml_dataset(
        test_path.to_str().unwrap(),
        -1f32,
        -1f32,
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


    let log_filename = "paper_vs_other";
    let log_filename = CString::new(log_filename).expect("CString::new failed");
    let paper_vs_other: *mut LinearModel = init_linear_model(input_count as u32, true, false);
    train_linear_model(
        paper_vs_other,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        0.001, 1_000_000,
        log_filename.as_ptr()
    );

    let model_path = "../../models/mlp_plastic_vs_other.json";
    let model_path_cstr = CString::new(model_path).expect("CString::new failed");
    save_linear_model(
        plastic_vs_other,
        model_path_cstr.as_ptr()
    );


    println!("metal");
    let metal1 = "../dataset/train/metal/metal_1025.jpg";
    let metal1_plastic_vs_other = predict_linear_model(plastic_vs_other,image_resize_vec(metal1, 32).as_mut_ptr());
    let metal1_metal_vs_other = predict_linear_model(metal_vs_other,image_resize_vec(metal1, 32).as_mut_ptr());
    let metal1_paper_vs_other = predict_linear_model(paper_vs_other,image_resize_vec(metal1, 32).as_mut_ptr());
    println!("plasticVsOther = {}", metal1_plastic_vs_other);
    println!("metalVsOther = {}", metal1_metal_vs_other);
    println!("paperVsOther = {}", metal1_paper_vs_other);


    println!("papaer");
    let paper1 = "../dataset/train/paper/paper_3044.jpg";
    let metal1_plastic_vs_other = predict_linear_model(plastic_vs_other,image_resize_vec(paper1, 32).as_mut_ptr());
    let metal1_metal_vs_other = predict_linear_model(metal_vs_other,image_resize_vec(paper1, 32).as_mut_ptr());
    let metal1_paper_vs_other = predict_linear_model(paper_vs_other,image_resize_vec(metal1, 32).as_mut_ptr());
    println!("plasticVsOther = {}", metal1_plastic_vs_other);
    println!("metalVsOther = {}", metal1_metal_vs_other);
    println!("paperVsOther = {}", metal1_paper_vs_other);

    println!("plastique");
    let plastique1 = "../dataset/train/plastic/plastic_6064.jpg";
    let metal1_plastic_vs_other = predict_linear_model(plastic_vs_other,image_resize_vec(plastique1, 32).as_mut_ptr());
    let metal1_metal_vs_other = predict_linear_model(metal_vs_other,image_resize_vec(plastique1, 32).as_mut_ptr());
    let metal1_paper_vs_other = predict_linear_model(paper_vs_other,image_resize_vec(plastique1, 32).as_mut_ptr());
    println!("plasticVsOther = {}", metal1_plastic_vs_other);
    println!("metalVsOther = {}", metal1_metal_vs_other);
    println!("paperVsOther = {}", metal1_paper_vs_other);
}
