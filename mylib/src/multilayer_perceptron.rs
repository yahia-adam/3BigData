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
use std::ffi::{c_char, c_float, CStr, CString};
use std::fs::File;
use std::io::Write;
use std::slice;
use ndarray::{Array1, Array2};

#[derive(Debug)]
pub struct MultiLayerPerceptron {
    d: Vec<usize>,
    w: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
    l: usize,
    is_classification: bool,
    pub loss: Vec<f32>,
    pub accuracy: Vec<f32>,
}

fn activation(x: f32, derivative: bool) -> f32 {
    if derivative {
        1.0 - x.tanh().powi(2)
    } else {
        x.tanh()
    }
}

#[no_mangle]
pub extern "C" fn init_mlp(
    npl: *const u32,
    npl_size: u32,
    is_classification: bool,
) -> *mut MultiLayerPerceptron {
    if npl.is_null() || npl_size == 0 {
        return std::ptr::null_mut();
    }

    let npl = unsafe { std::slice::from_raw_parts(npl, npl_size as usize) };

    let mut model = MultiLayerPerceptron {
        d: npl.iter().map(|&x| x as usize).collect(),
        w: Vec::with_capacity(npl.len()),
        x: Vec::with_capacity(npl.len()),
        deltas: Vec::with_capacity(npl.len()),
        l: npl.len() - 1,
        is_classification,
        loss: Vec::new(),
        accuracy: Vec::new(),
    };

    model.w = vec![Vec::new(); model.l + 1];
    for l in 1..=model.l {
        model.w[l] = (0..=model.d[l - 1])
            .map(|_| {
                let mut neuron_weights = vec![0.0];
                neuron_weights.extend((1..=model.d[l]).map(|_| {
                    let limit = (6.0 / (model.d[l - 1] + model.d[l]) as f32).sqrt();
                    rand::thread_rng().gen_range(-limit..limit)
                }));
                neuron_weights
            })
            .collect();
    }

    model.x = (0..=model.l)
        .map(|l| {
            let mut layer_x = vec![1.0];
            layer_x.extend(vec![0.0; model.d[l]]);
            layer_x
        })
        .collect();

    model.deltas = (0..=model.l).map(|l| vec![0.0; model.d[l] + 1]).collect();

    Box::into_raw(Box::new(model))
}

fn propagate(model: &mut MultiLayerPerceptron, sample_inputs: &[f32]) -> Result<(), String> {
    if sample_inputs.len() != model.d[0] {
        return Err("La taille des entrées ne correspond pas à la dimension de la couche d'entrée".to_string());
    }

    model.x[0][1..].copy_from_slice(sample_inputs);
    for l in 1..=model.l {
        let prev_layer = Array1::from_vec(model.x[l-1].clone());
        let weights = Array2::from_shape_vec((model.d[l-1]+1, model.d[l]+1), model.w[l].concat())
            .map_err(|e| e.to_string())?;
        
        let mut activations = weights.t().dot(&prev_layer);

        if l < model.l || model.is_classification {
            activations.mapv_inplace(|x| activation(x, false));
        }

        model.x[l][0] = 1.0;
        model.x[l][1..].copy_from_slice(&activations.as_slice().unwrap()[1..]);
    }
    Ok(())
}

fn backpropagate(model: &mut MultiLayerPerceptron, sample_expected_outputs: &[f32], learning_rate: f32) -> Result<(), String> {
    let last_layer_index = model.d.len() - 1;

    if sample_expected_outputs.len() != model.d[last_layer_index] {
        return Err("La taille des sorties attendues ne correspond pas à la dimension de la couche de sortie".to_string());
    }

    for j in 1..=model.d[last_layer_index] {
        let output_error = model.x[last_layer_index][j] - sample_expected_outputs[j - 1];
        model.deltas[last_layer_index][j] = if model.is_classification {
            output_error * activation(model.x[last_layer_index][j], true)
        } else {
            output_error
        };
    }

    for l in (1..last_layer_index).rev() {
        for i in 1..=model.d[l] {
            let error_sum: f32 = (1..=model.d[l + 1])
                .map(|j| model.w[l + 1][i][j] * model.deltas[l + 1][j])
                .sum();
            model.deltas[l][i] = error_sum * activation(model.x[l][i], true);
        }
    }

    for l in 1..=last_layer_index {
        for i in 0..=model.d[l - 1] {
            for j in 1..=model.d[l] {
                let weight_update = learning_rate * model.deltas[l][j] * model.x[l - 1][i];
                model.w[l][i][j] -= weight_update;
            }
        }
    }

    Ok(())
}

#[no_mangle]
pub extern "C" fn train_mlp(
    model: *mut MultiLayerPerceptron,
    x_train: *const c_float,
    y_train: *const c_float,
    train_data_size: u32,
    x_test: *const c_float,
    y_test: *const c_float,
    test_data_size: u32,
    learning_rate: f32,
    epochs: u32,
) -> bool {
    let model = unsafe { model.as_mut() }.ok_or("Invalid model pointer").unwrap();
    let input_col = model.d[0];
    let output_col = model.d[model.l];

    let x_train = unsafe { slice::from_raw_parts(x_train, (train_data_size as usize) * input_col) };
    let y_train = unsafe { slice::from_raw_parts(y_train, (train_data_size as usize) * output_col) };
    let x_test = unsafe { slice::from_raw_parts(x_test, (test_data_size as usize) * input_col) };
    let y_test = unsafe { slice::from_raw_parts(y_test, (test_data_size as usize) * output_col) };

    for epoch in 1..=epochs {
        let mut total_loss = 0.0;

        for _ in 0..train_data_size {
            let k = rand::thread_rng().gen_range(0..train_data_size) as usize;
            let sample_inputs = &x_train[k * input_col..(k + 1) * input_col];
            let sample_expected_outputs = &y_train[k * output_col..(k + 1) * output_col];

            if let Err(e) = propagate(model, sample_inputs) {
                eprintln!("Propagation error: {}", e);
                return false;
            }
            if let Err(e) = backpropagate(model, sample_expected_outputs, learning_rate) {
                eprintln!("Backpropagation error: {}", e);
                return false;
            }

            total_loss += calculate_loss(model, sample_expected_outputs);
        }

        let avg_loss = total_loss / train_data_size as f32;
        model.loss.push(avg_loss);

        if epoch % 10 == 0 {
            let accuracy = evaluate(model, x_test, y_test);
            model.accuracy.push(accuracy);
            println!("Epoch {}: Loss = {:.4}, Accuracy = {:.2}%", epoch, avg_loss, accuracy * 100.0);
        }
    }

    true
}

fn calculate_loss(model: &MultiLayerPerceptron, expected: &[f32]) -> f32 {
    model.x[model.l][1..].iter()
        .zip(expected.iter())
        .map(|(output, expected)| (output - expected).powi(2))
        .sum::<f32>() / expected.len() as f32
}

fn evaluate(model: &mut MultiLayerPerceptron, x_test: &[f32], y_test: &[f32]) -> f32 {
    let mut correct = 0;
    let test_size = y_test.len() / model.d[model.l];

    for i in 0..test_size {
        let inputs = &x_test[i * model.d[0]..(i + 1) * model.d[0]];
        let expected = &y_test[i * model.d[model.l]..(i + 1) * model.d[model.l]];
        
        if let Err(e) = propagate(model, inputs) {
            eprintln!("Evaluation error: {}", e);
            return 0.0;
        }

        let predicted = model.x[model.l][1..].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let actual = expected.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

        if (predicted - actual).abs() < 1e-6 {
            correct += 1;
        }
    }

    correct as f32 / test_size as f32
}

#[no_mangle]
pub extern "C" fn predict_mlp(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *const f32,
) -> *mut f32 {
    let model = match unsafe { model.as_mut() } {
        Some(m) => m,
        None => return std::ptr::null_mut(),
    };

    let input_size = model.d[0];
    let sample_inputs = unsafe { slice::from_raw_parts(sample_inputs, input_size) };

    if let Err(e) = propagate(model, sample_inputs) {
        eprintln!("Prediction error: {}", e);
        return std::ptr::null_mut();
    }

    let predictions = &model.x[model.l][1..];

    let output = predictions.to_vec();

    let _ptr = output.as_ptr() as *mut f32;
    Box::into_raw(output.into_boxed_slice()) as *mut f32
}

#[no_mangle]
pub extern "C" fn mlp_to_json(model: *const MultiLayerPerceptron) -> *mut c_char {
    let model_ref = unsafe { &*model };
    let json_obj = json!({
        "w": model_ref.w,
        "d": model_ref.d,
        "x": model_ref.x,
        "deltas": model_ref.deltas,
        "l": model_ref.l,
        "is_classification": model_ref.is_classification,
    });

    let json_str = serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    let c_str = CString::new(json_str).expect("Failed to convert string to CString");
    c_str.into_raw()
}

#[no_mangle]
pub extern "C" fn free_mlp(ptr: *mut MultiLayerPerceptron) {
    if !ptr.is_null() {
        unsafe {
            let _ = Box::from_raw(ptr);
        }
    }
}

#[no_mangle]
pub extern "C" fn save_mlp_model(model: *const MultiLayerPerceptron, filepath: *const c_char) -> bool {
    let path_str = match unsafe { CStr::from_ptr(filepath) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid filepath");
            return false;
        }
    };

    let model_str = unsafe {
        let json_ptr = mlp_to_json(model);
        let json_cstr = CStr::from_ptr(json_ptr);
        let result = json_cstr.to_str().unwrap_or("").to_string();
        let _ = CString::from_raw(json_ptr);
        result
    };

    match File::create(path_str) {
        Ok(mut file) => {
            if let Err(e) = write!(file, "{}", model_str) {
                eprintln!("Error writing to file: {}", e);
                false
            } else {
                println!("Model saved successfully to: {}", path_str);
                true
            }
        }
        Err(e) => {
            eprintln!("Error creating file: {}", e);
            false
        }
    }
}