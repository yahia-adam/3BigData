/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   linear_model.rs                                           :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */
use std::collections::HashMap;
use nalgebra::base::DMatrix;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use std::ffi::CString;
use std::ffi::{c_char, c_float, CStr};
use std::fs;
use std::fs::File;
use std::io::Write;
use ndarray_rand::rand::SeedableRng;
use tensorboard_rs::summary_writer::SummaryWriter;
use rand::prelude::StdRng;

const SEED: u64 = 42;
const SAVE_INTERVAL: u32 = 25;
const DISPLAY_INTERVAL: u32 = 10;

#[derive(Serialize, Deserialize)]
pub struct LinearModel {
    pub weights: Vec<f32>,
    pub weights_count: usize,
    pub is_classification: bool,
    pub is_multiclass: bool,
    pub train_loss: Vec<f64>,
    pub test_loss: Vec<f64>,
    pub train_accuracy: Vec<f32>,
    pub test_accuracy: Vec<f32>
}

#[no_mangle]
pub extern "C" fn init_linear_model(input_count: u32, is_classification: bool, is_multiclass: bool) -> *mut LinearModel {
    let mut rng = StdRng::seed_from_u64(SEED);

    let model: LinearModel = LinearModel {
        weights: vec![rng.gen_range(-1.0..1.0); (input_count + 1) as usize],
        weights_count: input_count as usize,
        is_classification,
        is_multiclass,
        train_loss: vec![],
        test_loss: vec![],
        train_accuracy: vec![],
        test_accuracy: vec![],
    };

    let boxed_model: Box<LinearModel> = Box::new(model);
    let leaked_boxed_model: *mut LinearModel = Box::leak(boxed_model);
    leaked_boxed_model.into()
}


#[no_mangle]
pub extern "C" fn train_linear_model(
    model: *mut LinearModel,
    x_train: *const c_float,
    y_train: *const c_float,
    train_data_size: u32,
    x_test: *const c_float,
    y_test: *const c_float,
    test_data_size: u32,
    learning_rate: f32,
    epochs: u32,
    log_filename: *const c_char,
    model_filename: *const c_char,
    display_loss: bool,
    display_tensorboad: bool,
    save_model: bool,
) {
    let model_ref: &mut LinearModel = unsafe { model.as_mut().unwrap() };

    let data_size: usize = train_data_size as usize;
    let features: &[f32] = unsafe { std::slice::from_raw_parts(x_train, data_size * model_ref.weights_count)};
    let mut features: Vec<f32> = features.to_vec().clone();
    let labels: &[f32] = unsafe { std::slice::from_raw_parts(y_train, data_size)};
    let labels: Vec<f32> = labels.to_vec().clone();

    let test_data_size = test_data_size as usize;
    let x_test: &[f32] = unsafe { std::slice::from_raw_parts(x_test, test_data_size * model_ref.weights_count)};
    let x_test: Vec<f32> = x_test.to_vec().clone();
    let y_test: &[f32] = unsafe { std::slice::from_raw_parts(y_test, test_data_size)};
    let y_test: Vec<f32> = y_test.to_vec().clone();

    let mut writer = SummaryWriter::new(&("../logs".to_string()));
    let mut map = HashMap::new();
    let mut rng = StdRng::seed_from_u64(SEED);

    let logfilename =  {
        let c_str = unsafe {CStr::from_ptr(log_filename)};
        let recipient = c_str.to_str().unwrap_or_else(|_| "no_logfilename");
        recipient
    };

    if model_ref.is_classification {
        let mut input: Vec<f32> = vec![0.0; model_ref.weights_count];
        let mut y_true:Vec<f32> = vec![];
        let mut y_pred:Vec<f32> = vec![];

        for epoch in 1..epochs + 1 {

            for _ in 0..(data_size - 1) {

                let i = rng.gen_range(0..train_data_size) as usize;

                let desired_output = labels[i];

                let mut m = 0;
                let start = i * model_ref.weights_count;

                for r in start..(start + model_ref.weights_count) {
                    input[m] = features[r];
                    m += 1;
                }

                let predicted_output = guess(model_ref, input.clone());
                let error = desired_output - predicted_output;
                y_true.push(desired_output);
                y_pred.push(predicted_output);
                for i in 1..(model_ref.weights_count + 1) {
                    model_ref.weights[i] += learning_rate * error * input[i - 1];
                }
                model_ref.weights[0] += learning_rate * error;

            }


            if display_loss || display_tensorboad {
                let train_loss = mse_epoch(&y_true, &y_pred);
                let train_accuracy = accuracy(&y_true, &y_pred);
                model_ref.train_loss.push(train_loss);
                model_ref.train_accuracy.push(train_accuracy);

                match evaluate(model_ref, &x_test, &y_test, test_data_size) {
                    Ok((test_accuracy, test_loss)) => {
                        if display_tensorboad {
                            map.insert("train_loss".to_string(), train_loss as f32);
                            map.insert("test_loss".to_string(), test_loss as f32);
                            map.insert("train_accuracy".to_string(), train_accuracy as f32);
                            map.insert("test_accuracy".to_string(), test_accuracy as f32);
                        }
                        if epoch % DISPLAY_INTERVAL == 0 {
                            if display_loss {
                                println!(
                                    "Epoch {}/{}: Loss = {:.4}, Accuracy = {:.4}%, Test_Loss = {:.4}, Test_Accuracy = {:.4}%",
                                    epoch,
                                    epochs,
                                    train_loss,
                                    train_accuracy * 100.0,
                                    test_loss,
                                    test_accuracy * 100.0,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Evaluation error: {}", e);
                        return;
                    }
                }
            }

            if save_model && epoch % SAVE_INTERVAL == 0 {
                save_linear_model(model, model_filename);
            }
            writer.add_scalars(&logfilename, &map, epoch as usize);
        }
        if display_tensorboad {
            writer.flush();
        }
        if save_model {
            save_linear_model(model, model_filename);
        }
    } else {
        let mut i = 0;
        while i < data_size {
            features.insert((i * model_ref.weights_count) + i, 1.0);
            i += 1;
        }
        let x = DMatrix::from_row_slice(data_size, model_ref.weights_count + 1, features.as_slice());
        let y =  DMatrix::from_row_slice(data_size, 1, labels.as_slice());
        let x_transpose = x.transpose();
        let xtx = x_transpose.clone() * x;
        match xtx.try_inverse() {
            Some(xtx_inv) => {
                let xty = x_transpose * y;
                let w = xtx_inv * xty;
                model_ref.weights = w.as_slice().to_vec();
            },
            None => {
                println!("La matrice X^T * X n'est pas inversible");
            }
        }
    }
}

fn evaluate(
    model: &mut LinearModel,
    x_test: &[f32],
    y_test: &[f32],
    test_data_size: usize,
) -> Result<(f32, f64), String> {
    let mut y_true:Vec<f32> = vec![];
    let mut y_pred:Vec<f32> = vec![];

    for i in 0..(test_data_size - 1) {
        let mut input: Vec<f32> = vec![0.0; model.weights_count];
        let desired_output = y_test[i];

        let mut m = 0;
        let start = i * model.weights_count;

        for r in start..(start + model.weights_count) {
            input[m] = x_test[r];
            m += 1;
        }

        let predicted_output = guess(model, input.clone());
        y_true.push(desired_output);
        y_pred.push(predicted_output);
    }
    let loss =  mse_epoch(&y_true, &y_pred);
    model.test_loss.push(loss);

    let accuracy = accuracy(&y_true, &y_pred);
    model.test_accuracy.push(accuracy);

    Ok((accuracy, loss))

}

#[no_mangle]
pub extern "C" fn predict_linear_model(model: *mut LinearModel, inputs: *mut f32) -> c_float {
    let model_ref: &mut LinearModel = unsafe { model.as_mut().unwrap() };
    let inputs: &[f32] = unsafe { std::slice::from_raw_parts(inputs, model_ref.weights_count) };
    guess(model_ref, inputs.to_vec())
}


pub fn guess(model: &mut LinearModel, inputs: Vec<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 1..(model.weights_count + 1) {
        sum += inputs[i - 1] * model.weights[i]
    }
    sum += model.weights[0];
    if model.is_classification && !model.is_multiclass {
        if sum >= 0.0 {
            sum = 1.0;
        } else {
            sum = -1.0;
        }
    }
    sum
}

#[no_mangle]
pub extern "C" fn to_json(model: *const LinearModel) -> *const c_char {
    let model: &LinearModel = unsafe { model.as_ref().unwrap() };
    let json_obj: serde_json::Value = json!({
        "weights": model.weights,
        "weights_count": model.weights_count,
        "is_classification": model.is_classification,
        "train_loss": model.train_loss,
        "test_loss": model.test_loss,
    });
    let json_str: String =
        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str: CString = CString::new(json_str).expect("Failed to convert string to CString");
    let ptr: *const c_char = c_str.into_raw();
    ptr
}


#[no_mangle]
pub extern "C" fn save_linear_model(model: *const LinearModel, filepath: *const c_char) {
    let path_cstr: &CStr = unsafe { CStr::from_ptr(filepath) };
    let path_str: &str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_e) => {
            println!("Unable to save model error converting filepath to str");
            return;
        }
    };
    let weights_str: &str = unsafe {
        let weights_ptr: *const c_char = to_json(model);
        CStr::from_ptr(weights_ptr).to_str().unwrap_or("")
    };

    if let Ok(mut file) = File::create(path_str) {
        if let Err(_) = write!(file, "{}", weights_str) {
            println!("Unable to save model error writing to file");
        }
    } else {
        println!("Unable to save model error creating file");
    }
    println!("Model saved successfully on: {}", path_str);
}


#[no_mangle]
pub extern "C" fn load_linear_model(json_str_ptr: *const c_char) -> *mut LinearModel {
    let json_str_cstr = unsafe { CStr::from_ptr(json_str_ptr) };
    let json_str: &str = match json_str_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            println!("Unable to load model: Failed to convert C string to Rust string");
            return std::ptr::null_mut();
        }
    };

    let model: LinearModel = {
        let data: String = fs::read_to_string(json_str).expect("LogRocket: error reading file");
        serde_json::from_str(&data).unwrap()
    };

    // Box the model and return a raw pointer to it
    let boxed_model: Box<LinearModel> = Box::new(model);
    Box::into_raw(boxed_model)
}


#[no_mangle]
pub extern "C" fn free_linear_model(model: *mut LinearModel) {
    unsafe {
        let _model= Box::from_raw(model);
    }
}

fn mse(y: f32, y_hat: f32) -> f64 {
    (y - y_hat).powi(2) as f64
}

fn mse_epoch(y_true: &[f32], y_pred: &[f32]) -> f64 {
    let n = y_true.len();
    let total_mse: f64 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y, &y_hat)| mse(y, y_hat))
        .sum();

    total_mse / n as f64
}

fn accuracy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let mut predicted_true = 0;
    for (y, y_hat) in y_true.iter().zip(y_pred.iter()) {
        if *y as i32 == *y_hat as i32 {
            predicted_true += 1;
        }
    }
    predicted_true as f32 / y_true.len() as f32
}

