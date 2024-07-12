mod linear_model_training;

use std::fs;
use std::path::{Path, PathBuf};

use mylib::{
    RadialBasisFunctionNetwork,
    init_rbf,
    train_rbf_regression,
    train_rbf_rosenblatt,
    predict_rbf_regression,
    predict_rbf_classification,
    free_rbf,
    save_rbf_model,
    rbf_to_json
};

use mylib::load_dataset;

const IMAGE_SIZE: u32 = 32;
const INPUT_DIM: usize = (IMAGE_SIZE * IMAGE_SIZE) as usize;
const CLUSTER_NUM: i32 = 3;
const GAMMA: f32 = 0.01;
const EPOCHS: i32 = 100;
const LEARNING_RATE: f32 = 0.01;

fn main() {
    let base_dir = PathBuf::from("/home/adam/esgi/3BigData/dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    println!("Chemin d'entraînement: {:?}", train_path);
    println!("Chemin de test: {:?}", test_path);

    let (train_images, mut train_labels) = load_dataset(train_path.to_str().unwrap());
    let (test_images, test_labels) = load_dataset(test_path.to_str().unwrap());

    let model = init_rbf(INPUT_DIM as i32, CLUSTER_NUM, GAMMA);

    train_rbf_rosenblatt(
        model,
        train_images.concat().as_mut_ptr(),
        train_labels.as_mut_ptr(),
        EPOCHS,
        LEARNING_RATE,
        INPUT_DIM as i32,
        train_images.len() as i32
    );

    let correct = test_images.iter().zip(test_labels.iter())
        .filter(|(img, &label)| {
            let mut img_copy = img.to_vec();
            let prediction = predict_rbf_classification(model, img_copy.as_mut_ptr());
            (prediction >= 0.0 && label >= 0.0) || (prediction < 0.0 && label < 0.0)
        })
        .count();

    let accuracy = correct as f32 / test_labels.len() as f32;
    println!("Précision du test: {:.2}%", accuracy * 100.0);

    // save_rbf_model(model, "model.json\0".as_ptr() as *const i8);
    free_rbf(model);
}