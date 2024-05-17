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
use std::ffi::{c_char, c_float, CStr};
use std::fs::File;
use std::io::Write;
use serde_json::{self, json};
use std::ffi::CString;
use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Serialize, Deserialize)]
pub struct LinearModel
{
    pub weights: Vec<f32>,
    pub weights_count: usize,
}

#[no_mangle]
pub extern "C" fn init_linear_model(input_count: u32) -> *mut LinearModel {
    let model: LinearModel = LinearModel {
        weights: vec![0.0; (input_count + 1) as usize],
        weights_count: input_count as usize,
    };

    let boxed_model: Box<LinearModel> = Box::new(model);
    let leaked_boxed_model: *mut LinearModel = Box::leak(boxed_model);
    leaked_boxed_model.into()
}

#[no_mangle]
pub extern "C" fn train_linear_model(model: *mut LinearModel, features: *const c_float, labels: *const c_float, data_size: u32, learning_rate: f32, epochs: u32) {
    let data_size: usize = data_size as usize;

    let model_ref: &mut LinearModel = unsafe {
        model.as_mut().unwrap()
    };
    let features: &[f32] = unsafe {
        std::slice::from_raw_parts(features, (data_size * model_ref.weights_count) as usize)
    };
    let labels: &[f32] = unsafe {
        std::slice::from_raw_parts(labels, data_size as usize)
    };

    let mut input: Vec<f32> = vec![0.0; model_ref.weights_count];

    for _ in 0..epochs {
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
            
            for i in 1..(model_ref.weights_count + 1) {
                model_ref.weights[i] += learning_rate * error * input[i - 1];
            }
            model_ref.weights[0] += learning_rate * error;
        }
    }
}

#[no_mangle]
pub extern "C" fn predict_linear_model(model: *mut LinearModel, inputs: *mut f32) -> c_float
{
    let model_ref: &mut LinearModel = unsafe {
        model.as_mut().unwrap()
    };
    let inputs: Vec<f32> =
        unsafe { Vec::from_raw_parts(inputs, model_ref.weights_count, model_ref.weights_count) };

    guess(model_ref , inputs)
}

pub fn guess(model: &mut LinearModel, inputs: Vec<f32>) -> f32
{
    let mut sum: f32 = 0.0;
    for i in 1..(model.weights_count + 1) {
        sum += inputs[i - 1] * model.weights[i]
    }
    sum += model.weights[0];
    sum
}

#[no_mangle]
pub extern "C" fn to_json(model: *const LinearModel) -> *const c_char {
    let model: &LinearModel = unsafe { model.as_ref().unwrap() };
    let json_obj: serde_json::Value = json!({
        "weights": model.weights,
    });
    let json_str: String = serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
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
    let _: &mut LinearModel = unsafe { model.as_mut().unwrap() };
}
