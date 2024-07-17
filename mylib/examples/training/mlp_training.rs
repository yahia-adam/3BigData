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

    println!("loading train dataset...");
    let (train_images, train_labels)  = load_dataset(
        train_path.to_str().unwrap(),
        -1f32,
        0f32,
        1f32
    );
    println!("loading test dataset...");
    let (test_images, test_labels) = load_dataset(
        test_path.to_str().unwrap(),
        -1f32,
        0f32,
        1f32
    );

    println!("finish loading dataset");
    let train_data_size = train_labels.len();
    let test_data_size = test_labels.len();
    let input_size = train_images[0].len();


    let train_images_flaten: Vec<f32> = train_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_train_ptr: *const f32 = Vec::leak(train_images_flaten.clone()).as_ptr();
    let y_train_ptr: *const f32 = Vec::leak(train_labels.clone()).as_ptr();

    let test_images_flaten: Vec<f32> = test_images.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_test_ptr: *const f32 = Vec::leak(test_images_flaten.clone()).as_ptr();
    let y_test_ptr: *const f32 = Vec::leak(test_labels.clone()).as_ptr();

    let mut npl: Vec<u32> = vec![input_size as u32, 64, 32, 3];
    let npl_size = npl.len();

    let model: *mut MultiLayerPerceptron = init_mlp(npl.as_mut_ptr(), npl_size as u32, true);

    println!("finis initing");
    train_mlp(
        model,
        x_train_ptr,
        y_train_ptr,
        train_data_size as u32,
        x_test_ptr,
        y_test_ptr,
        test_data_size as u32,
        0.5,
        100_000);
    
    let model_path = "../models/pmc_dataset.json";
    let model_path_cstr = CString::new(model_path).expect("CString::new failed");
    save_mlp_model(
        model,
        model_path_cstr.as_ptr()
    );

    println!("metal");
    let metal1 = "../dataset/train/metal/metal_1025.jpg";
    let res = predict_mlp(model, image_resize_vec(metal1, 32).as_mut_ptr());
    if !res.is_null() {
        let res: Vec<f32> = unsafe { Vec::from_raw_parts(res, 3, 3) };
        println!("paper: -1\nmetal: 0\nplastic: 1\n{:?}", res);
    }
}
 