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
    let y = vec![1f32, 1f32, -1f32, -1f32, -1f32, ];
    train(&model, &x, &y, &2.0, 5, 2);
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


fn train(model: &SVMModel, inputs: &Vec<Vec<f32>>, labels: &Vec<f32>, gamma: &f32, input_length: u32, dimensions: u32) {
    let total_input:usize = (input_length * dimensions) as usize;
    let dimensions = dimensions as usize;
    let input_length = input_length as usize;


    let mut big_matrix: Vec<Vec<f64>> = Vec::with_capacity(input_length);
    for i in 0..input_length {
        let row: Vec<f64> = (0..input_length)
            .map(|j| {
                let value = labels[i] as f32 * labels[j] as f32 * get_kernel(model, &inputs[i], &inputs[j]);
                value as f64
            })
            .collect();
        big_matrix.push(row);
    }

    println!("P: {:?}", big_matrix);

     let big_csc_matrix = CscMatrix::from(&big_matrix).into_upper_tri();
    println!("P csc: {:?}", big_csc_matrix);

    let mut q: Vec<f64> = vec![-1f64; input_length];
    // let mut q: Vec<f64> = Vec::with_capacity(input_length);
    // //q.extend(vec![0f64; input_length]);
    // q.extend(vec![*gamma as f64; input_length]);

    println!("q :{:?}", q);

    // let identity:OMatrix<f64, Dyn, Dyn> = DMatrix::identity(dimensions, dimensions);

    // let mut identity:Vec<Vec<f64>> = Vec::with_capacity(dimensions);
    // for i in 0..dimensions{
    //     let row:Vec<f64> = (0..dimensions).map(|j| {(i == j).into()}).collect();
    //     identity.push(row);
    // }
    // println!("id: {:?}", identity);


    let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; input_length]; (input_length * 2 + 1)];

    for i in 0..input_length {
        a_matrix[0][i] = labels[i] as f64;
        a_matrix[i + 1][i] = 1f64;
        a_matrix[i + input_length + 1][i] = -1f64;
    }
    println!("a :{:?}", a_matrix);

    let mut l: Vec<f64> = Vec::with_capacity((input_length * 2 + 1) as usize);
    l.extend(vec![-f64::INFINITY; input_length+1 ]);
    l.extend(vec![0f64; input_length]);
    println!("l: {:?}", l);

    let mut u: Vec<f64> = Vec::with_capacity((input_length * 2 + 1) as usize);
    u.extend(vec![1f64; input_length + 1]);
    u.extend(vec![f64::INFINITY; input_length as usize]);


    let settings = Settings::default().verbose(true);
    let mut problem = Problem::new(&big_csc_matrix, &*q, &a_matrix, &*l, &*u, &settings).expect("OSQP Setup Error");

    // Solve problem
    let result = problem.solve();

    // Print the solution
    println!("{:?}", result.x().expect("failed to solve problem"));
}
