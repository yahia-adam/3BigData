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

use nalgebra::DMatrix;
use osqp::{CscMatrix, Problem, Settings};

pub struct SVMModel {
    weight: f32,
    biais: f32,
    sample_size: u32,
    sample_len: u32,
    kernel: String,
    deg: i32,
}

fn main() {

    // Define problem data
    /*let P = &[[4.0, 1.0],
        [1.0, 2.0]];
    let q = &[1.0, 1.0];
    let A = &[[1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]];
    let l = &[1.0, 0.0, 0.0];
    let u = &[1.0, 0.7, 0.7];

    // Extract the upper triangular elements of `P`
    let P = CscMatrix::from(P);

    // Disable verbose output
    let settings = Settings::default()
        .verbose(false);

    // Create an OSQP problem
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");

    // Solve problem
    let result = prob.solve();

    // Print the solution
    println!("{:?}", result.x().expect("failed to solve problem"));*/


    let mut model = SVMModel {
        weight: 0.0,
        biais: 0.0,
        sample_size: 3,
        sample_len: 9,
        deg: 2,
        kernel: "linear".parse().unwrap(),
    };

    let x: Vec<Vec<f32>> = Vec::from([
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![2.0, 2.0],
        vec![4.0, 1.0],
        vec![4.0, 4.0],
    ]);
    let y = &[1, 1, -1, -1, -1, ];
    train(&model, &x, y, &2.0)
}

fn get_kernel(model: &SVMModel, xi: &Vec<f32>, xj: &Vec<f32>) -> f32 {
    if model.kernel == "linear" {
        //poduit scalaire
        xi.iter().zip(xj.iter()).map(|(i, j)| i * j).sum()
    } /*else if model.kernel == "poly" {
        f32::powi(1.0 + xi * xj, model.deg)
    } else if model.kernel == "rad" {
        expf(-f32::powi(xi, 2)) * expf(-f32::powi(xj, 2)) * expf(2.0 * xi * xj)
    }
    */else { 0.0 }
}

fn train(model: &SVMModel, inputs: &Vec<Vec<f32>>, labels: &[i32], gamma: &f32) {
    let mut big_matrix: Vec<Vec<f64>> = Vec::new();

    for i in 0..inputs.len() {
        big_matrix.push(Vec::new());
        for j in 0..inputs.len() {
            big_matrix[i].push((labels[i] as f32 * labels[j] as f32 * get_kernel(model, &inputs[i], &inputs[j])) as f64)
        }
    }

    println!("Big Matrix: {:?}", big_matrix);

    let p = CscMatrix::from(&big_matrix);

    let q = vec![0.0f32; model.sample_size as usize].append(&mut vec![*gamma; inputs.len()]);

    let identity = DMatrix::identity(model.sample_len as usize, model.sample_len as usize);
    let diag =
    //osqp::Problem::new

    println!("q :{:?}", q);

}
