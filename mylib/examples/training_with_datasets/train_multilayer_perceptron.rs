use mylib::{create_serialized_mlp_dataset, free_mlp, init_mlp, loads_serialized_mlp_dataset, train_mlp, MultiLayerPerceptron};
use std::ffi::CString;

const EPOCHS: u32 = 100;
const DIMENSIONS: &[&[u32]] = &[
    &[32, 32, 3],
    &[128, 64, 32, 3],
    &[256, 128, 64, 3],
    &[64, 64, 32, 3],
    &[128, 128, 64, 3],
];

fn main() {
    let (train_images, test_images, train_labels, test_labels);

    if let Err(e) = create_serialized_mlp_dataset("../dataset", "../serialized_datasets/anciens_dataset.bin") {
        println!("Failed to create serialized dataset: {}", e);
    } else {
        println!("yes");
    }

    // exit(0);
    match loads_serialized_mlp_dataset("../serialized_datasets/serialized_datasets.bin") {
        Ok((ti, tl, tei, tel)) => {
            println!("Dataset chargé avec succès !");
            train_images = ti;
            train_labels = tl;
            test_images = tei;
            test_labels = tel;
        }
        Err(e) => {
            eprintln!("Erreur lors du chargement du dataset : {}", e);
            std::process::exit(1);
        }
    }

    let train_data_size = train_labels.len();
    let test_data_size = test_labels.len();
    let input_size = train_images[0].len();

    let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    let x_train_ptr: *const f32 = train_images_flatten.as_ptr();
    let train_labels_flatten: Vec<f32> = train_labels.into_iter().flatten().collect();
    let y_train_ptr: *const f32 = train_labels_flatten.as_ptr();

    let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    let x_test_ptr: *const f32 = test_images_flatten.as_ptr();
    let test_label_flatten: Vec<f32> = test_labels.into_iter().flatten().collect();
    let y_test_ptr: *const f32 = test_label_flatten.as_ptr();

    let learning_rates = vec![0.001];

    for dim in DIMENSIONS {
        for &lr in &learning_rates {
            let mut npl: Vec<u32> = vec![input_size as u32];
            npl.extend_from_slice(dim);

            let npl_size = npl.len();
            let model: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl_size as u32, true);

            println!("Initialisation du modèle terminée pour la dimension {:?}", dim);

            let c_log_filename = CString::new(format!("3new_data{:?}lr={}epochs={}", dim, lr, EPOCHS)).expect("CString::new failed");
            let c_model_filename = CString::new(format!("../models/mlp/{:?}lr={}epochs{}.json", dim, lr, EPOCHS)).expect("CString::new failed");

            train_mlp(
                model,
                x_train_ptr,
                y_train_ptr,
                train_data_size as u32,
                x_test_ptr,
                y_test_ptr,
                test_data_size as u32,
                lr,
                EPOCHS,
                c_log_filename.as_ptr(),
                c_model_filename.as_ptr(),
                true,
                true,
                true,
            );
            free_mlp(model);

            println!("Entraînement terminé pour la dimension {:?} avec lr={}", dim, lr);
        }
    }
}