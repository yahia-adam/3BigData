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
use std::collections::HashMap;
use rand::Rng;
use serde_json::{self, json};
use std::ffi::c_char;
use std::ffi::c_float;
use std::ffi::CStr;
use std::ffi::CString;
use std::fs::File;
use std::io::Write;
use tensorboard_rs::summary_writer::SummaryWriter;

pub struct MultiLayerPerceptron {
    d: Vec<usize>,
    w: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
    l: usize,
    is_classification: bool,
    loss: Vec<f32>
}


#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn init_mlp(npl: *mut u32, npl_size: u32, is_classification: bool) -> *mut MultiLayerPerceptron {
    let npl = unsafe {
        std::slice::from_raw_parts(npl, npl_size as usize)
    };

    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        d: npl.into_iter().map(|x| *x as usize).collect(),
        w: vec![vec![vec![]]; npl.len()],
        x: vec![vec![]; npl.len()],
        deltas: vec![vec![]; npl.len()],
        l:  npl_size as usize - 1,
        is_classification: is_classification as bool,
        loss: vec![]
    };
    
    for l in 0..model.l + 1 {
        let layer_weights: Vec<Vec<f32>> = Vec::new();
        model.w[l] = layer_weights;
        if l == 0 {
            continue;
        }
        for _ in 0..model.d[l - 1] + 1 {
            let mut neuron_weights: Vec<f32> = Vec::new();
            for j in 0..model.d[l] + 1 {
                if j == 0 {
                    neuron_weights.push(0.0);
                } else {
                    neuron_weights.push(rand::thread_rng().gen_range(-1.0..1.0))
                }
            }
            model.w[l].push(neuron_weights);
        }
    }

    for l in 0..model.d.len() {
        let mut layer_x: Vec<f32> = Vec::new();
        let mut layer_delta: Vec<f32> = Vec::new();
        for j in 0..model.d[l] + 1 {
            layer_delta.push(0.0);
            if j == 0 {
                layer_x.push(1.0)
            } else {
                layer_x.push(0.0)
            }
        }
        model.x[l] = layer_x;
        model.deltas[l] = layer_delta;
    }

    let boxed_model: Box<MultiLayerPerceptron> = Box::new(model);
    Box::leak(boxed_model)
}


fn propagate(model: &mut MultiLayerPerceptron, sample_inputs: Vec<f32>) {
    for j in 0..sample_inputs.len() {
        model.x[0][j + 1] = sample_inputs[j];
    }
    for l in 1..model.l + 1 {
        for j in 1..model.d[l] + 1 {
            let mut total: f32 = 0.0;
            for i in 0..model.d[l - 1] + 1 {
                total += model.w[l][i][j] * model.x[l - 1][i];
            }
            if model.is_classification || l < model.d.len() - 1 {
                total = total.tanh();
            }
            model.x[l][j] = total;
        }
    }
}


fn backpropagate(
    model: &mut MultiLayerPerceptron,
    sample_expected_outputs: &[f32],
) {
    let last_layer_index: usize = model.d.len() - 1;

    for j in 1..model.d[last_layer_index] + 1 {
        model.deltas[last_layer_index][j] =
            model.x[last_layer_index][j] - &sample_expected_outputs[j - 1];
        if model.is_classification {
            model.deltas[last_layer_index][j] *= 1.0 - model.x[last_layer_index][j].powi(2)
        }
    }

    for l in (2..last_layer_index).rev() {
        for i in 1..model.d[l] + 1 {
            let mut total: f32 = 0.0;
            for j in 1..model.d[l] + 1 {
                total += model.w[l][i][j] * model.x[l - 1][i]
            }
            total *= 1.0 - model.x[l - 1][i].powi(2);
            model.deltas[l - 1][i] = total;
        }
    }
}


fn update_w(model: &mut MultiLayerPerceptron, alpha: f32) {
    for l in 1..model.d.len() {
        for i in 0..model.d[l - 1] + 1 {
            for j in 1..model.d[l] + 1 {
                model.w[l][i][j] -= alpha * model.x[l - 1][i] * model.deltas[l][j];
            }
        }
    }
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
    
    let input_col: usize = model_ref.d[0] as usize;
    let output_col: usize = model_ref.d[model_ref.l] as usize;
    let data_size: usize = data_size as usize;

    let inputs: Vec<f32> = unsafe { Vec::from_raw_parts(inputs, data_size * input_col, data_size * input_col) };

    let outputs: Vec<f32> =
        unsafe { Vec::from_raw_parts(outputs, data_size * output_col, data_size * output_col) };


    let mut writer = SummaryWriter::new(&("./logdir".to_string()));

    for n_iter in 0..nb_iteration {
        let mut map = HashMap::new();
        let k: usize = rand::thread_rng().gen_range(0..data_size);
        let sample_inputs: Vec<f32> = inputs[k * input_col..(k + 1) * input_col].to_vec();
        let sample_expected_outputs: Vec<f32> =
            outputs[k * output_col..(k + 1) * output_col].to_vec();

        propagate(model_ref, sample_inputs);
        backpropagate(model_ref, &sample_expected_outputs);
        update_w(model_ref, alpha);


        let mut mse: f32 = 0.0;
        for j in 1..model_ref.d[model_ref.d.len() - 1] + 1 {
            let error = model_ref.x[model_ref.d.len() - 1][j] - sample_expected_outputs[j - 1];
            mse += error.powi(2);
        }
        mse /= model_ref.d[model_ref.d.len() - 1] as f32;
        model_ref.loss.push(mse);
        map.insert("i".to_string(), n_iter as f32);
        map.insert("loss".to_string(), mse);
        writer.add_scalars("data/scalar_group", &map, n_iter as usize);


    }
    writer.flush();
}


#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn predict_mlp(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *mut f32,
) -> *mut f32 {
    let model_ref: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };

    let sample_inputs: Vec<f32> =
        unsafe { Vec::from_raw_parts(sample_inputs, model_ref.d[0], model_ref.d[0]) };

    propagate(model_ref, sample_inputs);

    let last_layer: usize = model_ref.d.len() - 1;
    let predictions: &[f32] = &model_ref.x[last_layer][1..];
    let cloned_predictions: Vec<f32> = predictions.to_vec();
    let result: &mut [f32] = Vec::leak(cloned_predictions);

    result.as_mut_ptr()
}


#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn mlp_to_json(model: *mut MultiLayerPerceptron) -> *mut c_char {
    let model_ref: &mut MultiLayerPerceptron = unsafe { model.as_mut().unwrap() };
    let json_obj: serde_json::Value = json!({
        "w": model_ref.w,
        "d": model_ref.d,
        "x": model_ref.x,
        "deltas" : model_ref.deltas,
        "l": model_ref.l,
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