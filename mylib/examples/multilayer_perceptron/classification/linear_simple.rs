use std::ffi::CString;
#[allow(unused_imports)]
use mylib::{free_mlp, init_mlp, predict_mlp, save_mlp_model, train_mlp, MultiLayerPerceptron};

fn main() {
    let x: Vec<Vec<f32>> = vec![vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 3.0]];
    let y: Vec<f32> = vec![1.0, -1.0, -1.0];

    let data_size = y.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let y_train: Vec<f32> = y.clone();

    let x_train_ptr: *const f32 = x_flaten.as_ptr();
    let y_train_ptr: *const f32 = y_train.as_ptr();

    let x_test_ptr: *const f32 = x_flaten.as_ptr();
    let y_test_ptr: *const f32 = y_train.as_ptr();

    let npl: Vec<u32> = vec![2, 1];
    let mlp: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl.len() as u32, true);


    let c_log_filename = CString::new("clf_linear_simple").expect("CString::new failed");
    let c_model_filename = CString::new("../models/examples/mlp/classification/linear_multiple.json").expect("CString::new failed");

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
            10_000,
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
                    let res: Vec<f32> = Vec::from_raw_parts(output, 1, 1);
                    println!("X: {:?}, Y: {:?} ---> MLP model: {:?}", x[i], y[i], res);
                    if (res[0] > 0.0 && y[i] > 0.0) || (res[0] <= 0.0 && y[i] <= 0.0) {
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
