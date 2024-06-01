/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   radical_basis_function_network.rs                         :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */


use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct RadicalBasisFunctionNetwork {
    weights : Vec<f32>,
    centers : Vec<Vec<f32>>,
    gamma : f32,
}

#[no_mangle]
pub extern "C" fn init_rbf(input_dim : i32, cluster_num : i32, gamma : f32) -> *mut RadicalBasisFunctionNetwork{
    let mut weights = Vec::with_capacity(cluster_num as usize);
    for _ in 0..cluster_num as usize{
        weights.push(0f32);
    }

    let mut centers = Vec::with_capacity(cluster_num as usize);
    for i in 0..cluster_num as usize{
        centers.push(Vec::with_capacity(input_dim as usize));
        for _ in 0..input_dim as usize{
            centers[i].push(0f32);
        }
    }

    let model = RadicalBasisFunctionNetwork {
        weights,
        centers,
        gamma,
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

pub fn euclid(x : &[f32], y : &[f32]) -> f32{
    let mut res = 0f32;
    for i in 0..(x.len()){
        res += (y[i] - x[i]).powi(2);
    }
    res.sqrt()
}

