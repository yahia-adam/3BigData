#[allow(unused_imports)]
use mylib::{free_mlp, init_mlp, predict_mlp, save_mlp_model, train_mlp, MultiLayerPerceptron};
use mylib::load_mlp_dataset;
use std::ffi::CString;
use std::path::PathBuf;

const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
// const DIM: &[u32] = &[512, 3];

// let dims = vec![
//     // 2. Variation du nombre de couches
//     vec![512, 3],                  // a. Une couche cachée
//     vec![512, 256, 3],             // b. Deux couches cachées
//     vec![512, 256, 128, 3],        // c. Trois couches cachées

//     // 3. Variation du nombre de neurones
//     vec![128, 64, 3],              // a. Peu de neurones
//     vec![1024, 512, 3],            // b. Beaucoup de neurones

//     // 4. Architecture symétrique
//     vec![512, 256, 512, 3],

//     // 5. Architecture avec goulot d'étranglement
//     vec![512, 64, 512, 3],

//     // 6. Test avec dropout (structure identique à 2b)
//     vec![512, 256, 3]
// ];

fn main() {
    let base_dir = PathBuf::from("../dataset");
    let train_path = base_dir.join("train");
    let test_path = base_dir.join("test");

    println!("Loading train dataset...");
    let (train_images, train_labels) = load_mlp_dataset(train_path.to_str().unwrap());

    println!("Loading test dataset...");
    let (test_images, test_labels) = load_mlp_dataset(test_path.to_str().unwrap());

    println!("Finished loading dataset");

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

    let dims = vec![
//     // 2. Variation du nombre de couches
    vec![512, 3],                  // a. Une couche cachée
    vec![512, 256, 3],             // b. Deux couches cachées
    vec![512, 256, 128, 3],        // c. Trois couches cachées
    ];

    for d in dims {
        let mut npl: Vec<u32> = vec![input_size as u32];
        npl.extend(d.clone());

        let npl_size = npl.len();
        let model: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl_size as u32, true);

        println!("Finished initializing model");

        let model_prameter: String = format!("Variation_couches:dim={:?}epoch={}lr={}", d, EPOCHS, LEARNING_RATE);
        let c_log_filename: CString =
            CString::new(model_prameter.clone()).expect("CString::new failed");
        let c_model_filename: CString =
            CString::new(format!("../models/dataset/mlp/{}.json", model_prameter))
                .expect("CString::new failed");

        train_mlp(
            model,
            x_train_ptr,
            y_train_ptr,
            train_data_size as u32,
            x_test_ptr,
            y_test_ptr,
            test_data_size as u32,
            LEARNING_RATE,
            EPOCHS,
            c_log_filename.as_ptr(),
            c_model_filename.clone().as_ptr(),
            true,
            true,
            true,
        );
        free_mlp(model);
    }
}
