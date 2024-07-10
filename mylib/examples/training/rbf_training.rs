use std::fs;
use std::path::{Path, PathBuf};


#[path = "../../src/radical_basis_function_network.rs"]
mod radical_basis_function_network;
use radical_basis_function_network::*;

const IMAGE_SIZE: u32 = 32;
const INPUT_DIM: usize = (IMAGE_SIZE * IMAGE_SIZE) as usize;
const CLUSTER_NUM: i32 = 3;
const GAMMA: f32 = 0.01;
const EPOCHS: i32 = 100;
const LEARNING_RATE: f32 = 0.01;

fn load_dataset(base_dir: &str) -> (Vec<Vec<f32>>, Vec<f32>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    let classes = ["metal", "paper", "plastic"];

    for class in classes.iter() {
        let class_dir = Path::new(base_dir).join(class);
        for entry in fs::read_dir(class_dir).expect("Erreur lors de la lecture du répertoire") {
            let path = entry.expect("Erreur lors de la lecture de l'entrée").path();
            if path.is_file() {
                if let Ok(img) = image::open(&path) {
                    let img = img.resize_exact(IMAGE_SIZE, IMAGE_SIZE, image::imageops::FilterType::Lanczos3);
                    let img_data: Vec<f32> = img.to_luma8().into_raw()
                        .into_iter()
                        .map(|p| p as f32 / 255.0)
                        .collect();

                    images.push(img_data);
                    labels.push(match *class {
                        "metal" => -1.0,
                        "paper" => 0.0,
                        "plastic" => 1.0,
                        _ => panic!("Classe inconnue"),
                    });
                } else {
                    eprintln!("Impossible d'ouvrir l'image: {:?}", path);
                }
            }
        }
    }
    (images, labels)
}

fn main() {
    let base_dir = PathBuf::from(r"C:\Users\csalhab\OneDrive\Online Sessions\3iabd1\projet annuel\3BigData\dataset");
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