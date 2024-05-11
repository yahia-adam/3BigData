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

use rand::Rng;
use serde_json::{self, json};

pub struct MultiLayerPerceptron {
    dimension: Vec<usize>,
    // vector of int represents the number of neurons in each layer or the number of layers
    weights: Vec<Vec<Vec<f32>>>,
    // un vecteur tridimensionnel, represents the weights of the neurons
    x: Vec<Vec<f32>>,
    // un vecteur de vecteurs, represents the entries of the MLP
    delta: Vec<Vec<f32>>, // un vecteur de vecteurs, represents errors / corrections to the MLP
}

#[no_mangle]
#[allow(dead_code)]
extern "C" fn init_mlp(npl: *mut usize, npl_size:usize) -> *mut MultiLayerPerceptron {
    let npl:Vec<usize>  = unsafe {
        Vec::from_raw_parts(npl, npl_size, npl_size)
    };

    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        dimension: npl.clone(),
        weights: vec![vec![vec![]]; npl.len()],
        x: vec![vec![]; npl.len()],
        delta: vec![vec![]; npl.len()],
    };

    for l in 1..model.dimension.len() {
        let mut layer_weights: Vec<Vec<f32>> = Vec::new();
        for _ in 0..model.dimension[l - 1] + 1 {
            let mut neuron_weights: Vec<f32> = Vec::new();
            // println!("{}", model.dimension[l] + 1);
            for j in 0..model.dimension[l] + 1 {
                // println!("{l}  {i}");
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

#[allow(dead_code)]
fn mlp_to_json(model: MultiLayerPerceptron) -> String {
    let json_obj: serde_json::Value = json!({
        "weights": model.weights,
        "dimension": model.dimension,
        "x": model.x,
        "delta" : model.delta,
    });

    let json_str: String =
        serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    json_str
}

#[no_mangle]
pub extern "C" fn free_mlp(
    model: *mut MultiLayerPerceptron,
) {
    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };
}

#[allow(dead_code)]
fn mlp_train(
    model: *mut MultiLayerPerceptron,
    inputs: *mut f32,
    outputs: *mut f32,
    input_size: usize,
    output_size: usize,
    data_size: usize,    
    alpha: f32, // learning rate
    nb_iteration: i32,
    is_classification: bool,
) {
    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };

    let inputs:Vec<f32>  = unsafe {
        Vec::from_raw_parts(inputs, data_size * input_size, data_size * input_size)
    };
   
    let outputs:Vec<f32>  = unsafe {
        Vec::from_raw_parts(outputs, data_size * output_size, data_size * output_size)
    };

    
    for _ in 0..nb_iteration {
        let k: usize = rand::thread_rng().gen_range(0..inputs.len());
        let sample_inputs: Vec<f32> = inputs[k-1*input_size..k*input_size].to_vec();
        let sample_expected_outputs: Vec<f32> = outputs[k-1*output_size..k*output_size].to_vec();

        propagate(model_ref, sample_inputs, is_classification);

        backpropagate(model_ref, &sample_expected_outputs, is_classification);

        update_weights(model_ref, alpha);
    }
}

fn backpropagate(
    model: &mut MultiLayerPerceptron,
    sample_expected_outputs: &[f32],
    is_classification: bool,
) {
    let last_layer_index: usize = model.dimension.len() - 1;

    for j in 1..model.dimension[last_layer_index] + 1 {
        let error: f32 = model.x[last_layer_index][j] - sample_expected_outputs[j - 1];
        model.delta[last_layer_index][j] = if is_classification {
            error * (1.0 - model.x[last_layer_index][j].powi(2)) // dérivée de tanh
        } else {
            error // Si c'est une régression, pas de fonction d'activation dans la couche de sortie
        };
    }

    for l in (1..last_layer_index).rev() {
        for i in 1..model.dimension[l] + 1 {
            let mut error: f32 = 0.0;
            for j in 1..model.dimension[l + 1] + 1 {
                error += model.weights[l + 1][i][j] * model.delta[l + 1][j];
            }
            model.delta[l][i] = error * (1.0 - model.x[l][i].powi(2)); // tanh' = 1 - tanh^2
        }
    }
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

pub fn propagate(
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

#[allow(dead_code)]
#[no_mangle]
pub extern "C" fn mlp_predict(
    model: *mut MultiLayerPerceptron,
    sample_inputs: *mut f32,
    sample_inputs_size: usize,
    is_classification: bool,
) -> *const f32 {

    let model_ref: &mut MultiLayerPerceptron = unsafe {
        model.as_mut().unwrap()
    };

    let sample_inputs:Vec<f32> = unsafe {
        Vec::from_raw_parts(sample_inputs, sample_inputs_size,sample_inputs_size)
    };
    
    propagate(model_ref, sample_inputs, is_classification);
    let res: &mut [f32] = Vec::leak( model_ref.x[model_ref.dimension.len() - 1][1..model_ref.x.len() - 1].to_vec());
    res.as_ptr()
}
