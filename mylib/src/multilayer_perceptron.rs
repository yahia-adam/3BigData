/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   multilayer_perceptron.rs                                  :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */


use rand::Rng;
use serde_json::{self, json};
use std::ffi::c_char;
use std::ffi::c_float;
use std::ffi::CStr;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use std::result;
use std::vec;

pub struct MultiLayerPerceptron {
    pub neurons_per_layer:Vec<usize>,
    pub is_classification: bool,
    pub weights: Vec<Vec<Vec<f32>>>,
    pub layer_nbr: usize,
    pub activation: Vec<Vec<f32>>,
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn init_mlp(neurons_per_layer: *mut u32, npl_size: u32, is_classification: bool) -> *mut MultiLayerPerceptron {
    let neurons_per_layer: Vec<usize> = unsafe {
        Vec::from_raw_parts(neurons_per_layer, npl_size as usize, npl_size as usize)
            .into_iter()
            .map(|x| x as usize)
            .collect()
    };

    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        neurons_per_layer,
        is_classification,
        weights: vec![],
        layer_nbr: (npl_size - 1) as usize,
        activation: vec![]
    };

    for l in 1..model.layer_nbr + 1 {
        let mut layer: Vec<Vec<f32>> = vec![];
        for _ in 0..(model.neurons_per_layer[l]) {
            let mut neural: Vec<f32> = vec![];
            neural.push(1 as f32);
            for _ in 0..model.neurons_per_layer[l - 1] {
                neural.push(rand::thread_rng().gen_range(-1.0..1.0))
            }
            layer.push(neural)
        }
        model.weights.push(layer);
    }

    model.activation.push(vec![]);
    for l in 1..model.layer_nbr + 1 {
        let mut layer :Vec<f32> = vec![];
        for _ in 0..(model.neurons_per_layer[l]) {
            layer.push(0 as f32);
        }
        model.activation.push(layer);
    }


    // println!("{:?}", model.activation);
    // println!("{:?}", model.weights);
    // propagate(&mut model, vec![1.0,2.0]);
    // println!("{:?}", model.activation);
    let boxedmodel: Box<MultiLayerPerceptron> = Box::new(model);
    Box::leak(boxedmodel)
}

fn propagate(model: &mut MultiLayerPerceptron, sample_inputs: Vec<f32>) {

    model.activation[0] = sample_inputs.clone();
    model.activation[0].insert(0, 1 as f32);

    for (l, layers) in model.weights.iter().enumerate() {
        for (n, neurals) in layers.iter().enumerate() {
            let weighted_sum: Vec<f32> = neurals.iter().zip(model.activation[l].iter()).map(|(&x, &y)| x * y).collect();
            model.activation[l+1][n] = weighted_sum.iter().sum();
        }
    }
}


fn backpropagate(
    model: &mut MultiLayerPerceptron,
    sample_expected_outputs: &[f32],
) {
   
}

fn update_w(model: &mut MultiLayerPerceptron, alpha: f32) {
    
}


#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn train_mlp(
    model: *mut MultiLayerPerceptron,
    inputs: *mut c_float,
    outputs: *mut c_float,
    data_size: u32,
    alpha: c_float,
    nb_iteration: u32,
) {
    let model_ref: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };
    
    let input_col: usize = model_ref.neurons_per_layer[0] as usize;
    let output_col: usize = model_ref.neurons_per_layer[model_ref.layer_nbr] as usize;
    let data_size: usize = data_size as usize;

    let inputs: Vec<f32> = unsafe { Vec::from_raw_parts(inputs, data_size * input_col, data_size * input_col) };

    let outputs: Vec<f32> =
        unsafe { Vec::from_raw_parts(outputs, data_size * output_col, data_size * output_col) };

    for _ in 0..nb_iteration {
        let k: usize = rand::thread_rng().gen_range(0..data_size);
        let sample_inputs: Vec<f32> = inputs[k * input_col..(k + 1) * input_col].to_vec();
        let sample_expected_outputs: Vec<f32> =
            outputs[k * output_col..(k + 1) * output_col].to_vec();

        propagate(model_ref, sample_inputs);
        backpropagate(model_ref, &sample_expected_outputs);
        update_w(model_ref, alpha);
    }
}


#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn predict_mlp(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *mut f32,
) -> *mut f32 {
    let model_ref: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };

    let sample_inputs: Vec<f32> =
        unsafe { Vec::from_raw_parts(sample_inputs, model_ref.neurons_per_layer[0], model_ref.neurons_per_layer[0])};

    propagate(model_ref, sample_inputs);

    let mut result: Vec<f32> = model_ref.activation[model_ref.layer_nbr + 1].clone();
    result.as_mut_ptr()
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn mlp_to_json(model: *mut MultiLayerPerceptron) -> *mut c_char {
    let model_ref: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };
    let json_obj: serde_json::Value = json!({
        "weights": model_ref.weights,
        "activation" : model_ref.activation,
        "neurons_per_layer": model_ref.neurons_per_layer,
        "layer_nbr": model_ref.layer_nbr,
        "is_classification": model_ref.is_classification,
    });

    let json_str: String =
        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str: CString = CString::new(json_str).expect("Failed to convert string to CString");
    let ptr: *mut c_char = c_str.into_raw();
    ptr
}


#[no_mangle]
pub extern "C" fn free_mlp(model: *mut MultiLayerPerceptron) {
    let _: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };
}

#[no_mangle]
pub extern "C" fn save_mlp_model(
    model: *mut MultiLayerPerceptron,
    filepath: *const std::ffi::c_char,
) {
    let path_cstr: &CStr = unsafe { CStr::from_ptr(filepath) };
    let path_str: &str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_e) => {
            println!("Unable to save model error converting filepath to str");
            return;
        }
    };

    let model_str: &str = unsafe {
        let str_model_ptr: *mut c_char = mlp_to_json(model);
        std::ffi::CStr::from_ptr(str_model_ptr)
            .to_str()
            .unwrap_or("")
    };

    if let Ok(mut file) = File::create(path_str) {
        if let Err(_) = write!(file, "{}", model_str) {
            println!("Unable to save model error writing to file");
        }
    } else {
        println!("Unable to save model error creating file");
    }
    println!("Model saved successfully on: {}", path_str);
}