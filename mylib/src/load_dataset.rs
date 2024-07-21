/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   lib.rs                                                    :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

use std::fs;
use std::path::{Path, PathBuf};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use zstd::Encoder;

const IMAGE_SIZE: u32 = 32;

pub fn loads_mlp_dataset(base_dir: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
                    let mut img_data = img.to_luma8().into_raw()
                        .into_iter()
                        .map(|p| p as f32 / 255.0)
                        .collect();
                    z_score_normalize(&mut img_data);
                    images.push(img_data);

                    labels.push(match *class {
                        "metal" => vec![1f32, -1f32, -1f32],
                        "paper" => vec![-1f32, 1f32, -1f32],
                        "plastic" => vec![-1f32, -1f32, 1f32],
                        _ => panic!("Classe inconnue"),
                    });
                } else {
                    eprintln!("Impossible d'ouvrir l'image: {:?}", path);
                }
            }
        }
    }
    shuffle_dataset(&mut images, &mut labels);
    (images, labels)
}

// mlp dataset
#[derive(Serialize, Deserialize)]
struct MLPDataset {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<Vec<f32>>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<Vec<f32>>,
}

pub fn create_serialized_mlp_dataset(base_dir: &str, output_file: &str) -> std::io::Result<()> {
    let base_dir = PathBuf::from(base_dir);
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    println!("Loading train dataset...");
    let (train_images, train_labels) = loads_mlp_dataset(train_path.to_str().unwrap());

    println!("Loading test dataset...");
    let (test_images, test_labels) = loads_mlp_dataset(test_path.to_str().unwrap());

    let dataset = MLPDataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
    };

    println!("Serializing and compressing dataset...");
    let file = File::create(output_file)?;
    let buf_writer = BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, 3)?; // Le niveau de compression peut être ajusté (0-21)
    bincode::serialize_into(&mut encoder, &dataset).unwrap();
    encoder.finish()?;

    println!("Dataset serialized, compressed, and saved to {}", output_file);
    Ok(())
}

pub fn loads_serialized_mlp_dataset(input_file: &str) -> std::io::Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    println!("Loading and decompressing serialized dataset from {}", input_file);
    let file = File::open(input_file)?;
    let buf_reader = BufReader::new(file);
    let decoder = zstd::Decoder::new(buf_reader)?;
    let dataset: MLPDataset = bincode::deserialize_from(decoder).unwrap();

    Ok((dataset.train_images, dataset.train_labels, dataset.test_images, dataset.test_labels))
}

// ml dataset
#[derive(Serialize, Deserialize)]
struct MLDataset {
    train_images: Vec<Vec<f32>>,
    train_labels: Vec<f32>,
    test_images: Vec<Vec<f32>>,
    test_labels: Vec<f32>,
}

pub fn load_ml_dataset(base_dir: &str, metal_label: f32, paper_label: f32, plastic_label: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
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
                    let mut img_data: Vec<f32> = img.to_luma8().into_raw()
                        .into_iter()
                        .map(|p| p as f32 / 255.0)
                        .collect();
                    z_score_normalize(&mut img_data);
                    images.push(img_data);

                    labels.push(match *class {
                        "metal" => metal_label,
                        "paper" => paper_label,
                        "plastic" => plastic_label,
                        _ => panic!("Classe inconnue"),
                    });
                } else {
                    eprintln!("Impossible d'ouvrir l'image: {:?}", path);
                }
            }
        }
    }
    shuffle_ml_dataset(&mut images, &mut labels);
    (images, labels)
}
pub fn create_serialized_ml_dataset(base_dir: &str, output_file: &str, metal_label: f32, paper_label: f32, plastic_label: f32) -> std::io::Result<()> {
    let base_dir = PathBuf::from(base_dir);
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    println!("Loading train dataset...");
    let (train_images, train_labels) = load_ml_dataset(train_path.to_str().unwrap(), metal_label, paper_label, plastic_label);

    println!("Loading test dataset...");
    let (test_images, test_labels) = load_ml_dataset(test_path.to_str().unwrap(), metal_label, paper_label, plastic_label);

    let dataset = MLDataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
    };

    println!("Serializing and compressing dataset...");
    let file = File::create(output_file)?;
    let buf_writer = BufWriter::new(file);
    let mut encoder = Encoder::new(buf_writer, 3)?;
    bincode::serialize_into(&mut encoder, &dataset)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    encoder.finish()?;

    println!("Dataset serialized, compressed, and saved to {}", output_file);
    Ok(())
}

pub fn load_serialized_ml_dataset(input_file: &str) -> std::io::Result<(Vec<Vec<f32>>, Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
    println!("Loading and decompressing serialized dataset from {}", input_file);
    let file = File::open(input_file)?;
    let buf_reader = BufReader::new(file);
    let decoder = zstd::Decoder::new(buf_reader)?;
    let dataset: MLDataset = bincode::deserialize_from(decoder)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    Ok((dataset.train_images, dataset.train_labels, dataset.test_images, dataset.test_labels))
}

//  process images
pub fn shuffle_ml_dataset(images: &mut Vec<Vec<f32>>, labels: &mut Vec<f32>) {
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..images.len()).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_images = Vec::with_capacity(images.len());
    let mut shuffled_labels = Vec::with_capacity(labels.len());

    for &i in &indices {
        shuffled_images.push(images[i].clone());
        shuffled_labels.push(labels[i]);
    }

    *images = shuffled_images;
    *labels = shuffled_labels;
}
pub fn shuffle_dataset(images: &mut Vec<Vec<f32>>, labels: &mut Vec<Vec<f32>>) {
    let mut rng = thread_rng();
    let mut indices: Vec<usize> = (0..images.len()).collect();
    indices.shuffle(&mut rng);

    let mut shuffled_images = Vec::with_capacity(images.len());
    let mut shuffled_labels = Vec::with_capacity(labels.len());

    for &i in &indices {
        shuffled_images.push(images[i].clone());
        shuffled_labels.push(labels[i].clone());
    }

    *images = shuffled_images;
    *labels = shuffled_labels;
}

pub fn image_resize_vec(filename: &str, image_size: u32) -> Vec<f32> {
    let path = PathBuf::from(filename);
    if path.is_file() {
        if let Ok(img) = image::open(&path) {
            let mut img= img.resize_exact(image_size, image_size, image::imageops::FilterType::Lanczos3)
                .to_luma8()
                .into_raw()
                .into_iter()
                .map(|p| p as f32 / 255.0)
                .collect();
            z_score_normalize(&mut img);
            img
        } else {
            eprintln!("Impossible d'ouvrir l'image: {:?}", path);
            vec![]
        }
    } else {
        eprintln!("Impossible de trouver le fichier: {:?}", path);
        vec![]
    }
}

fn z_score_normalize(data: &mut Vec<f32>) {
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();
    for value in data.iter_mut() {
        *value = (*value - mean) / std_dev;
    }
}
