/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   support_vector_machine.rs                                 :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{self, json};
use std::ffi::CString;
use std::ffi::{c_char, c_float, CStr};
use std::fs;
use std::fs::File;
use std::io::Write;

#[derive(Serialize, Deserialize)]
pub struct SupportVectorClassifier {
    pub weights: Vec<f32>,
    pub weights_count: usize,
}

#[no_mangle]
pub extern "C" fn init_svc(input_count: u32) -> *mut SupportVectorClassifier {
    let model: SupportVectorClassifier = SupportVectorClassifier {
        weights: vec![rand::thread_rng().gen_range(-1.0..1.0); (input_count + 1) as usize],
        weights_count: input_count as usize,
    };

    let boxed_model: Box<SupportVectorClassifier> = Box::new(model);
    let leaked_boxed_model: *mut SupportVectorClassifier = Box::leak(boxed_model);
    leaked_boxed_model.into()
}

#[no_mangle]
pub extern "C" fn train_svc(
    model: *mut SupportVectorClassifier,
    features: *const c_float,
    labels: *const c_float,
    data_size: u32,
    learning_rate: f32,
    epochs: u32,
) {
    let data_size: usize = data_size as usize;

    let model_ref: &mut SupportVectorClassifier = unsafe { model.as_mut().unwrap() };
    let features: &[f32] = unsafe {
        std::slice::from_raw_parts(features, (data_size * model_ref.weights_count) as usize)
    };
    let labels: &[f32] = unsafe { std::slice::from_raw_parts(labels, data_size as usize) };

    let features: Vec<f32> = features.to_vec().clone();
    let labels: Vec<f32> = labels.to_vec().clone();


}

#[no_mangle]
pub extern "C" fn predict_svc(
    model: *mut SupportVectorClassifier,
    inputs: *mut f32,
) -> c_float {
    let model_ref: &mut SupportVectorClassifier = unsafe { model.as_mut().unwrap() };
    let inputs: &[f32] = unsafe { std::slice::from_raw_parts(inputs, model_ref.weights_count) };
    guess(model_ref, inputs.to_vec())
}

pub fn guess(model: &mut SupportVectorClassifier, inputs: Vec<f32>) -> f32 {
    let mut sum: f32 = 0.0;
    for i in 1..(model.weights_count + 1) {
        sum += inputs[i - 1] * model.weights[i]
    }
    sum += model.weights[0];
    if sum >= 0.0 {
        sum = 1.0;
    } else {
        sum = -1.0;
    }
    sum
}

#[no_mangle]
pub extern "C" fn svc_to_json(model: *const SupportVectorClassifier) -> *const c_char {
    let model: &SupportVectorClassifier = unsafe { model.as_ref().unwrap() };
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
pub extern "C" fn save_svc(model: *const SupportVectorClassifier, filepath: *const std::ffi::c_char) {
    let path_cstr: &CStr = unsafe { CStr::from_ptr(filepath) };
    let path_str: &str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_e) => {
            println!("Unable to save model error converting filepath to str");
            return;
        }
    };
    let weights_str: &str = unsafe {
        let weights_ptr: *const c_char = svc_to_json(model);
        std::ffi::CStr::from_ptr(weights_ptr).to_str().unwrap_or("")
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
pub extern "C" fn load_svc(json_str_ptr: *const c_char) -> *mut SupportVectorClassifier {
    let json_str_cstr = unsafe { CStr::from_ptr(json_str_ptr) };
    let json_str: &str = match json_str_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            println!("Unable to load model: Failed to convert C string to Rust string");
            return std::ptr::null_mut();
        }
    };

    let model: SupportVectorClassifier = {
        let data: String = fs::read_to_string(json_str).expect("LogRocket: error reading file");
        serde_json::from_str(&data).unwrap()
    };

    // Box the model and return a raw pointer to it
    let boxed_model: Box<SupportVectorClassifier> = Box::new(model);
    Box::into_raw(boxed_model)
}

#[no_mangle]
pub extern "C" fn free_svc(model: *mut SupportVectorClassifier) {
    let _: &mut SupportVectorClassifier = unsafe { model.as_mut().unwrap() };
}
