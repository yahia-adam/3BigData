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
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde_json::{self, json};
use std::collections::HashMap;
use std::ffi::{c_char, c_float, CStr, CString};
use std::fs::File;
use std::io::Write;
use std::slice;
use tensorboard_rs::summary_writer::SummaryWriter;

const SEED: u64 = 42;
const SAVE_INTERVAL: u32 = 10;

#[derive(Debug)]
pub struct MultiLayerPerceptron {
    d: Vec<usize>,
    w: Vec<Vec<Vec<f32>>>,
    x: Vec<Vec<f32>>,
    deltas: Vec<Vec<f32>>,
    l: usize,
    is_classification: bool,
    rng: StdRng,
}

fn tanh_activation(x: f32, derivative: bool) -> f32 {
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
    let result = init_mlp_internal(npl, npl_size, is_classification);

    match result {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(e) => {
            eprintln!("Error initializing MLP: {}", e);
            std::ptr::null_mut()
        }
    }
}

fn init_mlp_internal(
    npl: *const u32,
    npl_size: u32,
    is_classification: bool,
) -> Result<MultiLayerPerceptron, String> {
    if npl.is_null() {
        return Err("Invalid input parameters".to_string());
    }
    if npl_size < 2 {
        return Err("Error: npl_size must be gratter than 2".to_string());
    }

    let npl = unsafe { slice::from_raw_parts(npl, npl_size as usize) };

    let mut model = MultiLayerPerceptron {
        d: npl.iter().map(|&x| x as usize).collect(),
        w: Vec::with_capacity(npl.len()),
        x: Vec::with_capacity(npl.len()),
        deltas: Vec::with_capacity(npl.len()),
        l: npl.len() - 1,
        is_classification,
        rng: StdRng::seed_from_u64(SEED),
    };

    model.w = vec![Vec::new(); model.l + 1];
    for l in 1..=model.l {
        model.w[l] = (0..=model.d[l - 1])
            .map(|_| {
                let mut neuron_weights = vec![0.0];
                neuron_weights.extend((1..=model.d[l]).map(|_| {
                    let limit = (6.0 / (model.d[l - 1] + model.d[l]) as f32).sqrt();
                    model.rng.gen_range(-limit..limit)
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

    Ok(model)
}

fn propagate(model: &mut MultiLayerPerceptron, sample_inputs: &[f32]) -> Result<(), String> {
    if sample_inputs.len() != model.d[0] {
        return Err(
            "La taille des entrées ne correspond pas à la dimension de la couche d'entrée"
                .to_string(),
        );
    }

    model.x[0][1..].copy_from_slice(sample_inputs);
    for l in 1..=model.l {
        let prev_layer = Array1::from_vec(model.x[l - 1].clone());
        let weights =
            Array2::from_shape_vec((model.d[l - 1] + 1, model.d[l] + 1), model.w[l].concat())
                .map_err(|e| e.to_string())?;

        let mut activations = weights.t().dot(&prev_layer);
        activations.mapv_inplace(|x| tanh_activation(x, false));

        model.x[l][0] = 1.0;
        model.x[l][1..].copy_from_slice(&activations.as_slice().unwrap()[1..]);
    }
    Ok(())
}

fn backpropagate(
    model: &mut MultiLayerPerceptron,
    sample_expected_outputs: &[f32],
    learning_rate: f32,
) -> Result<(), String> {
    let last_layer_index = model.d.len() - 1;

    if sample_expected_outputs.len() != model.d[last_layer_index] {
        return Err("La taille des sorties attendues ne correspond pas à la dimension de la couche de sortie".to_string());
    }

    for j in 1..=model.d[last_layer_index] {
        let output_error = model.x[last_layer_index][j] - sample_expected_outputs[j - 1];
        model.deltas[last_layer_index][j] = if model.is_classification {
            output_error * tanh_activation(model.x[last_layer_index][j], true)
        } else {
            output_error
        };
    }

    for l in (1..last_layer_index).rev() {
        for i in 1..=model.d[l] {
            let error_sum: f32 = (1..=model.d[l + 1])
                .map(|j| model.w[l + 1][i][j] * model.deltas[l + 1][j])
                .sum();
            model.deltas[l][i] = error_sum * tanh_activation(model.x[l][i], true);
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

fn get_multiclass_accuracy(model: &mut MultiLayerPerceptron, expected_output: &[f32]) -> f64 {
    let predicted_output = &model.x[model.l][1..];
    let expected_output_max_index = expected_output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0);

    let predicted_output_max_index: usize = predicted_output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0);

    if expected_output_max_index == predicted_output_max_index {
        1f64
    } else {
        0f64
    }
}

fn get_mse(model: &mut MultiLayerPerceptron, expected_output: &[f32]) -> f64 {
    model.x[model.l].iter().skip(1)
        .zip(expected_output)
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum::<f32>() as f64
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
    log_filename: *const c_char,
    model_filename: *const c_char,
) -> bool {
    if model.is_null()
        || x_train.is_null()
        || y_train.is_null()
        || x_test.is_null()
        || y_test.is_null()
    {
        eprintln!("Error: Null pointer passed to train_mlp");
        return false;
    }

    if epochs <= 0 {
        eprintln!("Error: Epochs must be positif");
        return false;
    }
    let model = unsafe { &mut *model };

    let input_dim = model.d[0];
    let output_dim = model.d[model.l];
    if train_data_size == 0 || test_data_size == 0 {
        eprintln!("Error: Data size cannot be zero");
        return false;
    }

    let logfilename = format!(
        "{}{}_epochs{}_lr={}",
        "data/mlp/",
        {
            let c_str = unsafe { CStr::from_ptr(log_filename) };
            let recipient = c_str.to_str().unwrap_or_else(|_| "_{}_{}_");
            recipient
        },
        epochs,
        learning_rate
    );

    let x_train = unsafe { slice::from_raw_parts(x_train, (train_data_size as usize) * input_dim) };
    let y_train =
        unsafe { slice::from_raw_parts(y_train, (train_data_size as usize) * output_dim) };
    let x_test = unsafe { slice::from_raw_parts(x_test, (test_data_size as usize) * input_dim) };
    let y_test = unsafe { slice::from_raw_parts(y_test, (test_data_size as usize) * output_dim) };

    if x_train.len() != train_data_size as usize * input_dim {
        eprintln!("Error: Inconsistent x_train dimensions");
        return false;
    }
    if y_train.len() != train_data_size as usize * output_dim {
        eprintln!("Error: Inconsistent y_train dimensions");
        return false;
    }
    if x_test.len() != test_data_size as usize * input_dim {
        eprintln!("Error: Inconsistent x_test dimensions");
        return false;
    }
    if y_test.len() != test_data_size as usize * output_dim {
        eprintln!("Error: Inconsistent y_test dimensions");
        return false;
    }

    if !(0.0..=1.0).contains(&learning_rate) {
        eprintln!("Error: Learning rate should be between 0 and 1");
        return false;
    }

    let mut writer = SummaryWriter::new(&("../logs".to_string()));
    let mut map = HashMap::new();

    for epoch in 1..=epochs {
        let mut train_accuracy: Vec<f64> = Vec::new();
        let mut train_loss: Vec<f64> = Vec::new();

        for _ in 1..=train_data_size {
            let k = model.rng.gen_range(0..train_data_size) as usize;
            let sample_inputs = &x_train[k * input_dim..(k + 1) * input_dim];
            let sample_expected_outputs = &y_train[k * output_dim..(k + 1) * output_dim];

            match propagate(model, sample_inputs) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Propagation error: {}", e);
                    return false;
                }
            }

            match backpropagate(model, sample_expected_outputs, learning_rate) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Backpropagation error: {}", e);
                    return false;
                }
            }
            train_loss.push(get_mse(model, sample_expected_outputs));
            train_accuracy.push(get_multiclass_accuracy(model, sample_expected_outputs));
        }

        let train_accuracy = train_accuracy.iter().filter(|&n| *n == 1f64).count() as f32
            / train_accuracy.len() as f32;
        let train_loss =
            train_loss.iter().sum::<f64>() / train_loss.len() as f64 * model.d[model.l] as f64;

        match evaluate(model, x_test, y_test) {
            Ok((test_accuracy, test_loss)) => {
                map.insert("train_loss".to_string(), train_loss as f32);
                //map.insert("train_accuracy".to_string(), train_accuracy);
                map.insert("test_loss".to_string(), test_loss as f32);
                //map.insert("test_accuracy".to_string(), test_accuracy);

                println!(
                    "Epoch {}/{}: Loss = {:.4}, Acuuracy = {:.4}%",
                    epoch,
                    epochs,
                    train_loss,
                    train_accuracy * 100.0
                );
            }
            Err(e) => {
                eprintln!("Evaluation error: {}", e);
                return false;
            }
        }

        writer.add_scalars(&logfilename, &map, epoch as usize);
        if epoch % SAVE_INTERVAL == 0 {
            save_mlp_model(model, model_filename);
        }
    }
    writer.flush();
    true
}

fn evaluate(
    model: &mut MultiLayerPerceptron,
    x_test: &[f32],
    y_test: &[f32],
) -> Result<(f32, f64), String> {
    let test_size = y_test.len() / model.d[model.l];

    if x_test.len() % model.d[0] != 0 {
        return Err(format!("La taille des données de test d'entrée n'est pas un multiple de la dimension d'entrée du modèle ({})", model.d[0]));
    }
    if y_test.len() % model.d[model.l] != 0 {
        return Err(format!("La taille des données de test de sortie n'est pas un multiple de la dimension de sortie du modèle ({})", model.d[model.l]));
    }

    let test_size_x = x_test.len() / model.d[0];
    let test_size_y = y_test.len() / model.d[model.l];
    if test_size_x != test_size_y {
        return Err(format!("Le nombre d'échantillons d'entrée ({}) ne correspond pas au nombre d'échantillons de sortie ({})", test_size_x, test_size_y));
    }

    let mut test_accuracy: Vec<f64> = Vec::new();
    let mut test_loss: Vec<f64> = Vec::new();

    for i in 0..test_size {
        let inputs = &x_test[i * model.d[0]..(i + 1) * model.d[0]];
        let expected_output = &y_test[i * model.d[model.l]..(i + 1) * model.d[model.l]];

        propagate(model, inputs).map_err(|e| {
            format!(
                "Erreur lors de la propagation de l'échantillon {}: {}",
                i, e
            )
        })?;

        test_loss.push(get_mse(model, expected_output));
        test_accuracy.push(get_multiclass_accuracy(model, expected_output));
    }

    let test_mean_accuracy =
        test_accuracy.iter().filter(|&n| *n == 1f64).count() as f32 / test_accuracy.len() as f32;
    let test_mean_loss =
        test_loss.iter().sum::<f64>() / test_loss.len() as f64 * model.d[model.l] as f64;

    Ok((test_mean_accuracy, test_mean_loss))
}

#[no_mangle]
pub extern "C" fn predict_mlp(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *const f32,
) -> *mut f32 {
    let model = match unsafe { model.as_mut() } {
        Some(m) => m,
        None => {
            eprintln!("Invalid model pointer");
            return std::ptr::null_mut();
        }
    };

    let input_size = model.d[0];
    let sample_inputs = unsafe { slice::from_raw_parts(sample_inputs, input_size) };

    match propagate(model, sample_inputs) {
        Ok(_) => {
            let predictions = &model.x[model.l][1..];
            let output = predictions.to_vec();
            Box::into_raw(output.into_boxed_slice()) as *mut f32
        }
        Err(e) => {
            eprintln!("Prediction error: {}", e);
            std::ptr::null_mut()
        }
    }
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

    let json_str = match serde_json::to_string_pretty(&json_obj) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("JSON serialization error: {}", e);
            return std::ptr::null_mut();
        }
    };

    match CString::new(json_str) {
        Ok(c_str) => c_str.into_raw(),
        Err(e) => {
            eprintln!("CString conversion error: {}", e);
            std::ptr::null_mut()
        }
    }
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
pub extern "C" fn save_mlp_model(
    model: *const MultiLayerPerceptron,
    filepath: *const c_char,
) -> bool {
    let path_str = match unsafe { CStr::from_ptr(filepath) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid filepath");
            return false;
        }
    };

    let model_str = unsafe {
        let json_ptr = mlp_to_json(model);
        if json_ptr.is_null() {
            eprintln!("Failed to convert model to JSON");
            return false;
        }
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
