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

use std::ffi::{c_char, CString};
use rand::Rng;
use serde_json::{self, json};
use crate::linear_model::{get_weights, LinearRegression};

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
pub extern "C" fn init_mlp(npl: Vec<usize>) -> MultiLayerPerceptron {
    let mut model: MultiLayerPerceptron = MultiLayerPerceptron {
        dimension: npl.clone(),
        weights: vec![vec![vec![]]; npl.len()],
        x: vec![vec![]; npl.len()],
        delta: vec![vec![]; npl.len()],
    };


    for l in 1..model.dimension.len() {
        let mut layer_weights = Vec::new();
        for i in 0..model.dimension[l - 1] + 1 {
            let mut neuron_weights = Vec::new();
            println!("{}", model.dimension[l] + 1);
            for j in 0..model.dimension[l] + 1 {
                println!("{l}  {i}");
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
        let mut layer_x = Vec::new();
        let mut layer_delta = Vec::new();
        for j in 0..model.dimension[l] + 1 {
            layer_delta.push(0.0);
            if j == 0 {
                layer_x.push(1.0)
            } else {
               layer_x.push(0.0)
            }
        }
        model.x[l]=layer_x;
        model.delta[l] = layer_delta;
    }

    model
}

#[no_mangle]
pub extern "C" fn mlp_to_json(model: MultiLayerPerceptron) -> String {
    let json_obj: serde_json::Value = json!({
        "weights": model.weights,
        "dimension": model.dimension,
        "x": model.x,
        "delta" : model.delta,
    });

    let json_str: String = serde_json::to_string_pretty(&json_obj).unwrap_or_else(|_| "".to_string());
    json_str
}