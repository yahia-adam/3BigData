#[path = "../src/radical_basis_function_network.rs"]
mod radical_basis_function_network;
use radical_basis_function_network::*;
use image::{GrayImage, ImageBuffer, DynamicImage};
use std::fs;

fn process_image(path: &str) -> Vec<f32> {
    let img = image::open(path).unwrap().into_luma8();
    let resized = image::imageops::resize(&img, 32, 32, image::imageops::FilterType::Lanczos3);
    resized.into_vec().iter().map(|&p| p as f32 / 255.0).collect()
}

fn load_dataset(dir: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() {
            let img_data = process_image(path.to_str().unwrap());
            images.push(img_data);

            labels.push(match path.parent().unwrap().file_name().unwrap().to_str().unwrap() {
                "metal" => -1.0,
                "paper" => 0.0,
                "plastic" => 1.0,
                _ => panic!("Unknown class"),
            });
        }
    }
    (images, labels)
}

fn main() {
    // Charger les données
    let (train_images, train_labels) = load_dataset("path/to/train/data");
    let (test_images, test_labels) = load_dataset("path/to/test/data");

    // Initialiser le modèle
    let input_dim = 32 * 32; // 1024 pour des images 32x32
    let cluster_num = 50; // à ajuster selon vos besoins
    let gamma = 0.1; // à ajuster
    let model = init_rbf(input_dim, cluster_num, gamma);

    // Entraîner le modèle
    train_rbf_rosenblatt(
        model,
        train_images.concat().as_mut_ptr(),
        train_labels.as_mut_ptr(),
        100,
        0.01,
        input_dim as i32,
        train_images.len() as i32
    );

    let mut correct = 0;
    for (img, label) in test_images.iter().zip(test_labels.iter()) {
        let prediction = predict_rbf_classification(model, img.as_mut_ptr());
        if (prediction >= 0.0 && *label >= 0.0) || (prediction < 0.0 && *label < 0.0) {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / test_labels.len() as f32;
    println!("Test accuracy: {:.2}%", accuracy * 100.0);

    save_rbf_model(model, "model.json\0".as_ptr() as *const i8);

    free_rbf(model);
}