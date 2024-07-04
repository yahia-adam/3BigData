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
use tensorboard_rs::summary_writer::SummaryWriter;

#[derive(Serialize, Deserialize)]
pub struct LinearModel {
    pub weights: Vec<f32>,
    pub weights_count: usize,
    pub is_classification: bool,
    pub loss: Vec<f32>,
}

#[no_mangle]
pub extern "C" fn init_linear_model(input_count: u32, is_classification: bool) -> *mut LinearModel {
    let model: LinearModel = LinearModel {
        weights: vec![rand::thread_rng().gen_range(-1.0..1.0); (input_count + 1) as usize],
        weights_count: input_count as usize,
        is_classification: is_classification as bool,
        loss: vec![]
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
    learning_rate: f32,
    epochs: u32,
) {
    let data_size: usize = train_data_size as usize;

    let model_ref: &mut LinearModel = unsafe { model.as_mut().unwrap() };
    let features: &[f32] = unsafe {
        std::slice::from_raw_parts(x_train, (data_size * model_ref.weights_count) as usize)
    };
    let labels: &[f32] = unsafe { std::slice::from_raw_parts(y_train, data_size as usize) };

    let mut features: Vec<f32> = features.to_vec().clone();
    let labels: Vec<f32> = labels.to_vec().clone();

    let mut writer = SummaryWriter::new(&("../logs".to_string()));
    let mut map = HashMap::new();

    if model_ref.is_classification {
        let mut input: Vec<f32> = vec![0.0; model_ref.weights_count];
        let mut y_true:Vec<f32> = vec![];
        let mut y_pred:Vec<f32> = vec![];
        for n_iter in 0..epochs {
            for i in 0..(data_size - 1) {
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
                if error > 0.001 || error < -0.001 {
                    for i in 1..(model_ref.weights_count + 1) {
                        model_ref.weights[i] += learning_rate * error * input[i - 1];
                    }
                    model_ref.weights[0] += learning_rate * error;
                }
            }
            let loss = mse_epoch(&y_true, &y_pred);
            model_ref.loss.push(loss);
            map.insert("loss".to_string(), loss);
            writer.add_scalars("data/linear_model", &map, n_iter as usize);
        }
        writer.flush();
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
    if model.is_classification {
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
    });
    let json_str: String =
        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str: CString = CString::new(json_str).expect("Failed to convert string to CString");
    let ptr: *const c_char = c_str.into_raw();
    ptr
}


#[no_mangle]
pub extern "C" fn save_linear_model(model: *const LinearModel, filepath: *const std::ffi::c_char) {
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
        std::ffi::CStr::from_ptr(weights_ptr).to_str().unwrap_or("")
    };

    if let Ok(mut file) = File::create(path_str) {
        // Write the weights string to the file
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

fn mse(y: f32, y_hat: f32) -> f32 {
    (y - y_hat).powi(2) as f32
}

fn mse_epoch(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let n = y_true.len();
    let total_mse: f32 = y_true.iter()
        .zip(y_pred.iter())
        .map(|(&y, &y_hat)| mse(y, y_hat))
        .sum();
    total_mse / n as f32
}