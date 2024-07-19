use std::ffi::CString;
#[allow(unused_imports)]
use mylib::{free_mlp, init_mlp, predict_mlp, save_mlp_model, train_mlp, MultiLayerPerceptron};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.25, 0.75],
    ];
    let y: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0],  // Classe 0
        vec![0.0, 1.0, 0.0],  // Classe 1
        vec![0.0, 1.0, 0.0],  // Classe 1
        vec![0.0, 0.0, 1.0],  // Classe 2
        vec![0.0, 0.0, 1.0],  // Classe 2
        vec![0.0, 1.0, 0.0],  // Classe 1
    ];

    let data_size = y.len();
    let x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let y_flatten: Vec<f32> = y.clone().into_iter().flatten().collect();
    let x_train_ptr: *const f32 = x_flatten.as_ptr();
    let y_train_ptr: *const f32 = y_flatten.as_ptr();
    let x_test_ptr: *const f32 = x_flatten.as_ptr();
    let y_test_ptr: *const f32 = y_flatten.as_ptr();
    let npl: Vec<u32> = vec![2, 3, 3];  // 2 entrées, une couche cachée de 3 neurones, 3 sorties
    let mlp: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl.len() as u32, true);


    let c_log_filename = CString::new("clf_multi_linear_3_classes").expect("CString::new failed");
    let c_model_filename = CString::new("../models/examples/mlp/classification/multi_linear_3_classes.json").expect("CString::new failed");

    unsafe {
        let success = train_mlp(
            mlp,
            x_train_ptr,
            y_train_ptr,
            data_size as u32,
            x_test_ptr,
            y_test_ptr,
            data_size as u32,
            0.001,
            50_000,
            c_log_filename.as_ptr(),
            c_model_filename.as_ptr(),
            false,
            false,
            false,
        );

        if success {
            let mut correct = 0;
            for i in 0..data_size {
                let input_ptr: *const f32 = x[i].as_ptr();
                let output: *mut f32 = predict_mlp(mlp, input_ptr);
                if !output.is_null() {
                    let res: Vec<f32> = Vec::from_raw_parts(output, 3, 3);
                    println!("X: {:?}, Y: {:?} ---> MLP model: {:?}", x[i], y[i], res);
                    
                    let pred_class = res.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                    let true_class = y[i].iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                    
                    if pred_class == true_class {
                        correct += 1;
                    }
                }
            }
            let accuracy = correct as f32 / data_size as f32;
            println!("Accuracy: {:.2}%", accuracy * 100.0);
            // let model_file_name = "model.json";
            // let model_file_name = CString::new(model_file_name).expect("CString::new failed");
            // save_mlp_model(mlp, model_file_name.as_ptr());
        } else {
            println!("Training failed.");
        }
        free_mlp(mlp);
    }
}