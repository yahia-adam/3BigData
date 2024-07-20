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

use std::ffi::c_float;

use itertools::Itertools;
use osqp::{CscMatrix, Problem, Settings};
use crate::MultiLayerPerceptron;

pub struct SVMModel {
    dimensions: u32,
    weight: Vec<f32>,
    biais: f32,
    kernel: String,
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn init_svm(dimensions: u32, kernel: u32) -> *mut SVMModel {
    let model: SVMModel = SVMModel {
        dimensions,
        weight: Vec::new(),
        biais: 1f32,
        kernel: "linear".parse().unwrap(),
    };

    let boxed_model: Box<SVMModel> = Box::new(model);
    let leaked_boxed_model: *mut SVMModel = Box::leak(boxed_model);
    leaked_boxed_model.into()
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

fn normalize_data(data: &mut Vec<Vec<f32>>) {
    let features = data[0].len();
    for feature in 0..features {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for sample in data.iter() {
            min = min.min(sample[feature]);
            max = max.max(sample[feature]);
        }
        for sample in data.iter_mut() {
            sample[feature] = (sample[feature] - min) / (max - min);
        }
    }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn train_svm(model_pointer: *mut SVMModel, inputs_pointer: *mut c_float, labels_pointer: *mut c_float, input_length: u32, c: f32) {
    let model: &mut SVMModel = unsafe { model_pointer.as_mut().unwrap() };

    let dimensions: usize = model.dimensions as usize;
    let input_length: usize = input_length as usize;

    let flat_input = unsafe { Vec::from_raw_parts(inputs_pointer, dimensions * input_length, dimensions * input_length) };
    let inputs: Vec<Vec<f32>> = flat_input.chunks(dimensions).map(|c| c.to_vec()).collect();

    let labels: Vec<c_float> = unsafe { Vec::from_raw_parts(labels_pointer, input_length, input_length) };


    let mut big_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; dimensions + input_length + 1];
    for i in 0..dimensions {
        big_matrix[i][i] = 1f64;
    }

    //println!("P: {:?}", big_matrix);

    let big_csc_matrix = CscMatrix::from(&big_matrix).into_upper_tri();
    println!("P csc: {:?}", big_csc_matrix);

    let mut q: Vec<f64> = vec![0f64; dimensions + 1 + input_length];
    for i in 0..input_length {
        q[dimensions + 1 + i] = 1f64;
    }
    //println!("q :{:?}", q);

    let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; 2 * input_length];
    for row in 0..input_length {
        for col in 0..dimensions {
            a_matrix[row][col] = (labels[row] * inputs[row][col]) as f64
        }
        a_matrix[row][dimensions] = labels[row] as f64;
        a_matrix[row][row + dimensions + 1] = -1f64;
        a_matrix[row + input_length][row + dimensions + 1] = 1f64;
    }
    let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; 2 * input_length];
    for row in 0..input_length {
        for col in 0..dimensions {
            a_matrix[row][col] = (labels[row] * inputs[row][col]) as f64;
        }
        a_matrix[row][dimensions] = labels[row] as f64;
        a_matrix[row][row + dimensions + 1] = -1f64;
        a_matrix[row + input_length][row + dimensions + 1] = 1f64;
    }


    //println!("a :{:?}", a_matrix);
    let a_csc_matrix = CscMatrix::from(&a_matrix);
    println!("a_csc :{:?}", a_csc_matrix);

    let mut l: Vec<f64> = Vec::new();
    l.extend(vec![1f64; input_length]);
    l.extend(vec![0f64; input_length]);
    println!("l: {:?}", l);

    let mut u: Vec<f64> = vec![f64::INFINITY; 2 * input_length];
    println!("u: {:?}", u);

    let settings = Settings::default().verbose(true);
    let mut problem = Problem::new(&big_csc_matrix, &*q, &a_matrix, &*l, &*u, &settings).expect("OSQP Setup Error");

    // Solve problem
    let result = problem.solve();
    let alphas = result.x().expect("failed to solve problem");

    // Print the solution
    println!("alphas {:?}", alphas);


    let w: Vec<f32> = (0..dimensions)
        .map(
            |dim|
            alphas.iter()
                .zip(&labels)
                .zip(&inputs)
                .map(
                    |((alpha, label), input)|
                    *alpha as f32 * label * input[dim])
                .sum::<f32>())
        .collect();

    println!("weights: {:?}", w);


    let mut bias = 0f32;
    let mut sv_count = 0;
    for i in 0..input_length {
        if alphas[i].abs() > 1e-3 && alphas[i].abs() < c as f64 {
            bias += labels[i] - inputs[i].iter().zip(&w).map(|(x, w)| x * w).sum::<f32>();
            sv_count += 1;
        }
    }
    if sv_count > 0 {
        bias /= sv_count as f32;
    } else {
        println!("Warning: No support vectors found. The model may not be well-fitted.");
    }

    println!("bias: {} with {} sv", bias, sv_count);
    model.biais = bias;
    model.weight = w;
}

pub fn mse_svm(expected: &Vec<f32>, prediction: &Vec<f32>) -> f32 {
    let error: f32 = expected.iter().zip(prediction).map(|(exp, pred)| (exp - pred).powi(2)).sum::<f32>();
    let result: f32 = error / expected.len() as f32;

    result
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn predict_svm(model_pointer: *mut SVMModel, inputs_pointer: *mut c_float) -> c_float {
    let model: &mut SVMModel = unsafe { model_pointer.as_mut().unwrap() };
    let inputs: Vec<c_float> = unsafe { std::slice::from_raw_parts(inputs_pointer, model.dimensions as usize).to_vec() };

    let result: f32 = inputs.iter().zip(&model.weight).map(|(i, w)| i * w).sum::<f32>() + model.biais;

     println!("Input: {:?}, Raw score: {}", inputs,  result);
    result
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn free_svm(model_pointer: *mut SVMModel){
    let _: &mut SVMModel = unsafe { model_pointer.as_mut().unwrap() };
}