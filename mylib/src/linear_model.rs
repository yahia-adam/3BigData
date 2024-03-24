/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   linear_model.rs                                           :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 19:38:54 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */
use std::ffi::{c_char, CStr};
use std::fs::File;
use std::io::Write;
use serde_json::{self, json};
use std::ffi::CString;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct LinearRegression
{
    pub weights: Vec<f32>,
}

#[no_mangle]
pub extern "C" fn new() -> *mut LinearRegression {
    let model: LinearRegression = LinearRegression {
        weights: vec![],
    };

    let boxed_model: Box<LinearRegression> = Box::new(model);
    let leaked_boxed_model: *mut LinearRegression = Box::leak(boxed_model);
    leaked_boxed_model.into()
}

pub fn get_weights(model: *const LinearRegression) -> Vec<f32>
{
    let mode: &LinearRegression = unsafe{
        model.as_ref().unwrap()
    };
    mode.weights.clone()
}

pub fn set_weights(model: *mut LinearRegression, weights: Vec<f32>) {
    let model_ref: &mut LinearRegression = unsafe {
        model.as_mut().unwrap()
    };
    model_ref.weights = weights;
}

pub fn guess(model: *const LinearRegression, inputs: Vec<f32>) -> f32
{
    let model: &LinearRegression = unsafe{
        model.as_ref().unwrap()
    };

    let mut sum: f32 = 0.0;
    for (p,i) in model.weights.iter().skip(1).zip(inputs) {
        sum = sum + i * *p;
    }
    sum = sum + model.weights[0];
    sum
}

#[no_mangle]
pub extern "C" fn predict(model: *const LinearRegression, inputs: *const f32, input_size: usize,) -> f32
{
    let inputs: &[f32] = unsafe {
        std::slice::from_raw_parts(inputs, input_size)
    };
    let model: &LinearRegression = unsafe{
        model.as_ref().unwrap()
    };

    let mut sum: f32 = 0.0;
    for (p,i) in model.weights.iter().skip(1).zip(inputs) {
        sum = sum + i * *p;
    }
    sum = sum + model.weights[0];
    sum
}

#[no_mangle]
pub extern "C" fn to_json(model: *const LinearRegression) -> *const c_char {
    let model: &LinearRegression = unsafe { model.as_ref().unwrap() };
    let weights: Vec<f32> = get_weights(model);
    let json_obj: serde_json::Value = json!({
        "weights": weights
    });
    let json_str: String = serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str: CString = CString::new(json_str).expect("Failed to convert string to CString");
    let ptr: *const c_char = c_str.into_raw();
    ptr
}

#[no_mangle]
pub extern "C" fn save_model(model: *const LinearRegression, filepath: *const std::ffi::c_char) {
    let path_cstr: &CStr = unsafe { CStr::from_ptr(filepath) };
    let path_str: &str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_e) => {
            println!("Unaible to save model error converting filepath to str");
            return;
        }
    };

    // Call to_string to get the weights as a string
    let weights_str: &str = unsafe {
        let weights_ptr: *const c_char = to_json(model);
        std::ffi::CStr::from_ptr(weights_ptr).to_str().unwrap_or("")
    };

    if let Ok(mut file) = File::create(path_str) {
        // Write the weights string to the file
        if let Err(_) = write!(file, "{}", weights_str) {
            println!("Unaible to save model error writing to file");
        }
    } else {
        println!("Unaible to save model error creating file");
    }
    println!("Model saved successfuly on: {}", path_str);
}


#[no_mangle]
pub extern "C" fn load_model(json_str_ptr: *const c_char) -> *mut LinearRegression {
    let json_str_cstr = unsafe { CStr::from_ptr(json_str_ptr) };
    let json_str: &str = match json_str_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            println!("Unable to load model: Failed to convert C string to Rust string");
            return std::ptr::null_mut();
        }
    };

    let model: LinearRegression = {
        let data: String = fs::read_to_string(json_str).expect("LogRocket: error reading file");
        serde_json::from_str(&data).unwrap()
    };

    // Box the model and return a raw pointer to it
    let boxed_model = Box::new(model);
    Box::into_raw(boxed_model)
}


#[no_mangle]
pub extern "C" fn fit(model: *mut LinearRegression, labels: *const f32, label_row: usize, label_col: usize, features: *const f32, feature_size: usize, learning_rate: f32, epochs: u32) {
    let labels: &[f32] = unsafe {
        std::slice::from_raw_parts(labels, (label_col * label_row) as usize)
    };
    let features: &[f32] = unsafe {
        std::slice::from_raw_parts(features, feature_size as usize)
    };
    let model_ref: &mut LinearRegression = unsafe {
        model.as_mut().unwrap()
    };
    set_weights(model, vec![0.0; label_row + 1]);
    let mut epochs = epochs;
    let mut time = 0;
    let data_size = if label_row < feature_size {label_row} else {feature_size};
    while epochs != 0 {
        let mut count_label = 0;
        for count_feature in 0..data_size {
            let desired_output = features[count_feature];
            let mut input: Vec<f32> = vec![];
            for j in 0..label_col {
                let elem = labels[count_label + j];
                input.push(elem);
            }
            count_label  = count_label + label_col;
            let predicted_output = guess(model, input.clone());
            let mut weights: Vec<f32> = vec![0.0];
            for j in 0..label_col {
                weights.push(model_ref.weights[j+1] + learning_rate * (desired_output - predicted_output) * input[j]);
            }
            weights[0] = model_ref.weights[0] + learning_rate * (desired_output - predicted_output);
            set_weights(model_ref, weights);
            time = time + 1;
        }
        epochs = epochs - 1;
    }
}
