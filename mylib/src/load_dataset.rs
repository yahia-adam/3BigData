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


pub fn shuffle_dataset(images: &mut Vec<Vec<f32>>, labels: &mut Vec<f32>) {
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

pub fn image_resize_vec(filename: &str, image_size: u32) -> Vec<f32> {
    let path = PathBuf::from(filename);
    if path.is_file() {
        if let Ok(img) = image::open(&path) {
            img.resize_exact(image_size, image_size, image::imageops::FilterType::Lanczos3)
                .to_luma8()
                .into_raw()
                .into_iter()
                .map(|p| p as f32 / 255.0)
                .collect()
        } else {
            eprintln!("Impossible d'ouvrir l'image: {:?}", path);
            vec![]
        }
    } else {
        eprintln!("Impossible de trouver le fichier: {:?}", path);
        vec![]
    }
}


pub fn load_dataset(base_dir: &str, metal_label: f32, paper_label: f32, plastic_label: f32) -> (Vec<Vec<f32>>, Vec<f32>) {
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
    shuffle_dataset(&mut images, &mut labels);
    (images, labels)
}
