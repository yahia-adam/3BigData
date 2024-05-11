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

fn init_mlp(npl: Vec<usize>) -> MultiLayerPerceptron {
    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        dimension: npl.clone(),
        weights: vec![vec![vec![]]; npl.len()],
        x: vec![vec![]; npl.len()],
        delta: vec![vec![]; npl.len()],
    };

    for l in 1..model.dimension.len() {
        let mut layer_weights: Vec<Vec<f32>> = Vec::new();
        for i in 0..model.dimension[l - 1] + 1 {
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

    model
}

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

fn mlp_train(
    mut model: MultiLayerPerceptron,
    all_sample_inputs: Vec<Vec<f32>>,
    all_sample_outputs: Vec<Vec<f32>>,
    alpha: f32, // learning rate
    nb_iteration: i32,
    is_classification: bool,
) {
    for _ in 0..nb_iteration {
        let k: usize = rand::thread_rng().gen_range(0..all_sample_inputs.len());
        let sample_inputs: Vec<f32> = all_sample_inputs[k].clone();
        let sample_expected_outputs: Vec<f32> = all_sample_outputs[k].clone();

        model = propagate(model, sample_inputs, is_classification);

        backpropagate(&mut model, &sample_expected_outputs, is_classification);

        update_weights(&mut model, alpha);
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
    mut model: MultiLayerPerceptron,
    sample_inputs: Vec<f32>,
    is_classification: bool,
) -> MultiLayerPerceptron {
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

    return model;
}

pub fn predict(
    model: MultiLayerPerceptron,
    sample_inputs: Vec<f32>,
    is_classification: bool,
) -> Vec<f32> {
    let mlp_model: MultiLayerPerceptron = propagate(model, sample_inputs, is_classification);
    return mlp_model.x[mlp_model.dimension.len() - 1][1..mlp_model.x.len() - 1].to_vec();
}
