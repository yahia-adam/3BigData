/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   multilayer_perceptron.rs                                  :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 14:20:22 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 14:20:22 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::fs::File;
use rand::Rng;
use serde_json::{self, json};
use std::io::Write;
pub struct MultiLayerPerceptron {
    dimension: Vec<usize>,
    weights: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    delta: Vec<Vec<f32>>,
    l: usize
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn init_mlp(npl: *mut usize, npl_size:usize) -> *mut MultiLayerPerceptron {
    let npl:Vec<usize>  = unsafe {
        Vec::from_raw_parts(npl, npl_size, npl_size)
    };

    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        dimension: npl.clone(),
        weights: vec![vec![vec![]]; npl.len()],
        x: vec![vec![]; npl.len()],
        delta: vec![vec![]; npl.len()],
        l: npl.len() - 1,
    };

    for l in 1..model.dimension.len() {
        let mut layer_weights: Vec<Vec<f32>> = Vec::new();
        for _ in 0..model.dimension[l - 1] + 1 {
            let mut neuron_weights: Vec<f32> = Vec::new();
            for j in 0..model.dimension[l] + 1 {
                if j == 0 {
                    neuron_weights.push(0.0);
                } else {
                    neuron_weights.push(rand::thread_rng().gen_range(-1.0..1.0))
                }
            }
            layer_weights.push(neuron_weights);
        }
        model.weights[l] = layer_weights;
    }

    for l in 0..model.dimension.len() {
        let mut layer_x: Vec<f32> = Vec::new();
        let mut layer_delta: Vec<f32> = Vec::new();
        for j in 0..model.dimension[l] + 1 {
            layer_delta.push(0.0);
            if j == 0 {
                layer_x.push(1.0)
            } else {
                layer_x.push(0.0)
            }
        }
        model.x[l] = layer_x;
        model.delta[l] = layer_delta;
    }

    let boxed_model: Box<MultiLayerPerceptron> = Box::new(model);
    Box::leak(boxed_model)
}

fn update_weights(model: &mut MultiLayerPerceptron, alpha: f32) {
    for l in 1..model.dimension.len() {
        for i in 0..model.dimension[l - 1] + 1 {
            for j in 1..model.dimension[l] + 1 {
                model.weights[l][i][j] -= alpha * model.x[l - 1][i] * model.delta[l][j];
            }
        }
    }
}

fn propagate(
    model: &mut MultiLayerPerceptron,
    sample_inputs: Vec<f32>,
    is_classification: bool,
) {

    for j in 0..sample_inputs.len() {
        model.x[0][j + 1] = sample_inputs[j];
    }

    for l in 1..model.dimension.len() {
        for j in 1..model.dimension[l] {
            let mut total: f32 = 0.0;

            for i in 0..model.dimension[l - 1] + 1 {
                total += model.weights[l][i][j] * model.x[l - 1][i];
            }

            if is_classification || l < model.dimension.len() {
                total = total.tanh()
            }

            model.x[l][j] = total;
        }
    }
}

fn backpropagate(
    model: &mut MultiLayerPerceptron,
    sample_expected_outputs: &[f32],
    is_classification: bool,
) {

    for j in 1..model.dimension[model.l] + 1 {
        let error: f32 = model.x[model.l][j] - sample_expected_outputs[j - 1];
        model.delta[model.l][j] = if is_classification {
            error * (1.0 - model.x[model.l][j].powi(2))
        } else {
            error
        };
    }

    for l in (1..model.l).rev() {
        for i in 1..model.dimension[l] + 1 {
            let mut error: f32 = 0.0;
            for j in 1..model.dimension[l + 1] + 1 {
                error += model.weights[l + 1][i][j] * model.delta[l + 1][j];
            }
            model.delta[l][i] = error * (1.0 - model.x[l][i].powi(2));
        }
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn train_mlp(
    model: *mut MultiLayerPerceptron,
    inputs: *mut f32,
    outputs: *mut f32,
    input_col: usize,
    output_col: usize,
    row: usize,    
    alpha: f32,
    nb_iteration: i32,
    is_classification: bool,
) {
    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };

    let inputs:Vec<f32>  = unsafe {
        Vec::from_raw_parts(inputs, row * input_col, row * input_col)
    };
   
    let outputs:Vec<f32>  = unsafe {
        Vec::from_raw_parts(outputs, row * output_col, row * output_col)
    };
    
    for _ in 0..nb_iteration {
        let k: usize = rand::thread_rng().gen_range(0..row);
        let sample_inputs: Vec<f32> = inputs[k * input_col..(k + 1) * input_col].to_vec();
        let sample_expected_outputs: Vec<f32> = outputs[k * output_col..(k + 1) * output_col].to_vec();
        
        propagate(model_ref, sample_inputs, is_classification);
        backpropagate(model_ref, &sample_expected_outputs, is_classification);
        update_weights(model_ref, alpha);
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn predict_mlp(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *mut f32,
    sample_inputs_size: usize,
    is_classification: bool,
) -> *mut f32 {

    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };

    let sample_inputs:Vec<f32> = unsafe {
        Vec::from_raw_parts(sample_inputs, sample_inputs_size,sample_inputs_size)
    };
    
    propagate(model_ref, sample_inputs, is_classification);
    let res: &mut [f32] = Vec::leak( model_ref.x[model_ref.dimension.len() - 1][1..model_ref.x.len() - 1].to_vec());
    res.as_mut_ptr()
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn mlp_to_json(model: *mut MultiLayerPerceptron) ->  *mut c_char {
    
    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };
    let json_obj: serde_json::Value = json!({
        "weights": model_ref.weights,
        "dimension": model_ref.dimension,
        "x": model_ref.x,
        "delta" : model_ref.delta,
        "l": model_ref.l,
    });

    let json_str: String = serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str: CString = CString::new(json_str).expect("Failed to convert string to CString");
    let ptr: *mut c_char = c_str.into_raw();
    ptr
}

#[no_mangle]
pub extern "C" fn free_mlp(
    model: *mut MultiLayerPerceptron,
) {
    let _: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };
}

#[no_mangle]
pub extern "C" fn save_mlp_model(model: *mut MultiLayerPerceptron, filepath: *const std::ffi::c_char) {
    let path_cstr: &CStr = unsafe { CStr::from_ptr(filepath) };
    let path_str: &str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_e) => {
            println!("Unaible to save model error converting filepath to str");
            return;
        }
    };

    let model_str: &str = unsafe {
        let str_model_ptr: *mut c_char = mlp_to_json(model);
        std::ffi::CStr::from_ptr(str_model_ptr).to_str().unwrap_or("")
    };

    if let Ok(mut file) = File::create(path_str) {
        // Write the weights string to the file
        if let Err(_) = write!(file, "{}", model_str) {
            println!("Unaible to save model error writing to file");
        }
    } else {
        println!("Unaible to save model error creating file");
    }
    println!("Model saved successfuly on: {}", path_str);
}
