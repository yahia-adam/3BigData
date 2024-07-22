use mylib::{free_mlp, init_mlp, train_mlp, create_serialized_mlp_dataset, load_dataset, MultiLayerPerceptron};
use std::ffi::CString;

// const LEARNING_RATE: f32 = 0.001;
const EPOCHS: u32 = 100;
const DIM: &[u32] = &[512, 3];


// PS:  il faut cree les dossier suivant dans experimentation:
    // - EXP_TITLE
            // - models
            // - logs
const EXP_TITLE: &str = "Learning_rate";

fn main() {
    // if let Err(e) = create_serialized_mlp_dataset("../dataset", "../serialized_dataset.bin") {
    //     println!("Failed to create serialized dataset: {}", e);
    // }
    // let (train_images, train_labels, test_images, test_labels);
    // match loads_mlp_dataset("../serialized_dataset.bin") {
    //     Ok((ti, tl, tei, tel)) => {
    //         println!("Dataset chargé avec succès !");
    //         train_images = ti;
    //         train_labels = tl;
    //         test_images = tei;
    //         test_labels = tel;
    //     },
    //     Err(e) => {
    //         eprintln!("Erreur lors du chargement du dataset : {}", e);
    //         std::process::exit(1);
    //     }
    // }


    // let train_data_size = train_labels.len();
    // let test_data_size = test_labels.len();
    // let input_size = train_images[0].len();

    // let train_images_flatten: Vec<f32> = train_images.into_iter().flatten().collect();
    // let x_train_ptr: *const f32 = train_images_flatten.as_ptr();
    // let train_labels_flatten: Vec<f32> = train_labels.into_iter().flatten().collect();
    // let y_train_ptr: *const f32 = train_labels_flatten.as_ptr();

    // let test_images_flatten: Vec<f32> = test_images.into_iter().flatten().collect();
    // let x_test_ptr: *const f32 = test_images_flatten.as_ptr();
    // let test_label_flatten: Vec<f32> = test_labels.into_iter().flatten().collect();
    // let y_test_ptr: *const f32 = test_label_flatten.as_ptr();

    // let learning_rate = vec![
    //     // Learning rate
    //     0.1, 0.001, 0.0001, 0.00001,
    // ];

    // for lr in learning_rate {
    //     let mut npl: Vec<u32> = vec![input_size as u32];
    //     npl.extend(DIM);

    //     let npl_size = npl.len();
    //     let model: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl_size as u32, true);

    //     println!("Finished initializing model");

    //     let model_prameter: String =
    //         format!("dim={:?}epoch={}lr={}", DIM, EPOCHS, lr);

    //     let c_model_path: CString = CString::new(format!("../experiences/{}/models/mlp/{}.json", EXP_TITLE, model_prameter))
    //     .expect("CString::new failed");

    //     let c_model_log_path: CString = CString::new(format!("../experiences/{}/logs/mlp/{}", EXP_TITLE, model_prameter))
    //     .expect("CString::new failed");

    //     train_mlp(
    //         model,
    //         x_train_ptr,
    //         y_train_ptr,
    //         train_data_size as u32,
    //         x_test_ptr,
    //         y_test_ptr,
    //         test_data_size as u32,
    //         lr,
    //         EPOCHS,
    //         c_model_log_path.as_ptr(),
    //         c_model_path.clone().as_ptr(),
    //         true,
    //         true,
    //         true,
    //     );
    //     free_mlp(model);
    // }
}
