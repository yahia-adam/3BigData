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

const IMAGE_SIZE: u32 = 32;


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

pub fn load_dataset(base_dir: &str) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
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
