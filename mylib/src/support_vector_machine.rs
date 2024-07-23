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

use std::ffi::{c_char, c_float, CStr, CString};
use std::fs::File;
use std::io::Read;
use std::slice;
use std::io::Write;
use itertools::Itertools;
use libm::expf;
use osqp::{CscMatrix, Problem, Settings};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
pub struct SVMModel {
    dimensions: u32,
    weight: Vec<f32>,
    biais: f32,
    kernel: u32,
    kernel_value: f32,
    support_vectors: Vec<Vec<f32>>,
    support_labels: Vec<f32>,
    alphas: Vec<f32>,
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn init_svm(dimensions: u32, kernel: u32, kernel_value: f32) -> *mut SVMModel {
    let model: SVMModel = SVMModel {
        dimensions,
        weight: Vec::new(),
        biais: 1f32,
        kernel,
        kernel_value,
        support_vectors: vec![],
        support_labels: vec![],
        alphas: vec![],
    };

    let boxed_model: Box<SVMModel> = Box::new(model);
    let leaked_boxed_model: *mut SVMModel = Box::leak(boxed_model);
    leaked_boxed_model.into()
}

fn get_kernel(model: &SVMModel, xi: &Vec<f32>, xj: &Vec<f32>) -> f32 {
    if model.kernel == 1 {  //poduit scalaire
        xi.iter().zip(xj.iter()).map(|(i, j)| i * j).sum()
    } else if model.kernel == 2 { //Polynome de degrès = kernel_value
        xi.iter().zip(xj).map(|(i, j)| f32::powi(1f32 + i * j, model.kernel_value as i32)).sum()
    } else if model.kernel == 3 { //RBF avec gamma = kernel_value
        let squared_distance: f32 = xi.iter().zip(xj).map(|(i, j)| (i - j).powi(2)).sum();
        expf(-model.kernel_value * squared_distance)
    } else { 0.0 }
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn train_svm(model_pointer: *mut SVMModel, inputs_pointer: *const f32, labels_pointer: *const c_float, input_length: u32, c: f32, epsilon: f32) {
    let model: &mut SVMModel = unsafe { model_pointer.as_mut().unwrap() };

    let dimensions: usize = model.dimensions as usize;
    let input_length: usize = input_length as usize;

    let flat_input = unsafe { slice::from_raw_parts(inputs_pointer, dimensions * input_length) };
    let inputs: Vec<Vec<f32>> = flat_input.chunks(dimensions).map(|c| c.to_vec()).collect();

    let labels: Vec<c_float> = unsafe { slice::from_raw_parts(labels_pointer, input_length) }.to_vec();


    let mut big_matrix: Vec<Vec<f64>> =
    if model.kernel == 1 {
        let mut big_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; dimensions + input_length + 1];
        for i in 0..dimensions {
            big_matrix[i][i] = 1f64;
        }

        big_matrix
    } else {
        let mut big_matrix: Vec<Vec<f64>> = vec![vec![0f64; input_length + 1]; input_length + 1];
        for i in 0..input_length {
            for j in 0..input_length {
                let kernel_value = get_kernel(model, &inputs[i], &inputs[j]);
                big_matrix[i][j] = (labels[i] * labels[j] * kernel_value) as f64;
            }
        }

        big_matrix
    };

    for i in 0..big_matrix.len() {
        big_matrix[i][i] += 1e-6;
    }



    //println!("P: {:?}", big_matrix);

    let big_csc_matrix = CscMatrix::from(&big_matrix).into_upper_tri();
   // println!("P csc: {:?}", big_csc_matrix);


    let a_matrix: Vec<Vec<f64>> =
    if model.kernel == 1 {
        let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; 2 * input_length];
        for row in 0..input_length {
            for col in 0..dimensions {

                a_matrix[row][col] = (labels[row] * inputs[row][col]) as f64
            }
            a_matrix[row][dimensions] = labels[row] as f64;
            a_matrix[row][row + dimensions + 1] = -1f64;
            a_matrix[row + input_length][row + dimensions + 1] = 1f64;
        }

        a_matrix
    } else {
        let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; input_length + 1]; 2 * input_length];
        for row in 0..input_length {
            a_matrix[row][row] = labels[row] as f64;
            a_matrix[row][input_length] = labels[row] as f64;
            a_matrix[row + input_length][row] = 1f64;
        }

        a_matrix
    };

   // println!("a :{:?}", a_matrix);
    let a_csc_matrix = CscMatrix::from(&a_matrix);
    //println!("a_csc :{:?}", a_csc_matrix);

    let (q, l, u) = if model.kernel == 1 {
        let mut q = vec![0f64; dimensions + 1 + input_length];
        for i in 0..input_length {
            q[dimensions + 1 + i] = 1f64;
        }

        let mut l: Vec<f64> = Vec::new();
        l.extend(vec![1f64; input_length]);
        l.extend(vec![0f64; input_length]);

        let mut u: Vec<f64> = vec![f64::INFINITY; 2 * input_length];

        (q, l, u)
    }
    else {
        let mut q = vec![0f64; input_length + 1];
        for i in 0..input_length {
            q[i] = -1f64;
        }

        let l: Vec<f64> = vec![0f64; 2 * input_length];

        let mut u: Vec<f64> = vec![0f64; 2 * input_length];
        for i in 0..input_length {
            u[i] = c as f64;
        }
        for i in input_length..(2 * input_length) {
            u[i] = f64::INFINITY;
        }
        (q, l, u)
    };

//    println!("q :{:?}", q);
  //  println!("l: {:?}", l);
    //println!("u: {:?}", u);

    println!("Starting to solve...");
    let settings = Settings::default().verbose(true);
    let mut problem = Problem::new(&big_csc_matrix, &*q, &a_matrix, &*l, &*u, &settings).expect("OSQP Setup Error");

    // Solve problem
    let result = problem.solve();
    let alphas: Vec<f32> = result.x().expect("failed to solve problem").iter().map(|x| *x as f32).collect();

    // Print the solution
    //println!("alphas {:?}", alphas);


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

    //println!("weights: {:?}", w);


    let mut bias = 0f32;
    let mut sv_count = 0;
    for i in 0..input_length {
        if alphas[i].abs() > epsilon && alphas[i].abs() < c - epsilon {
            bias += labels[i] - inputs[i].iter().zip(&w).map(|(x, w)| x * w).sum::<f32>();
            sv_count += 1;
        }
    }
    if sv_count > 0 {
        bias /= sv_count as f32;
    } else {
        println!("Warning: No support vectors found. The model may not be well-fitted.");
    }

    //println!("bias: {} with {} sv", bias, sv_count);
    model.biais = bias;
    model.weight = w;

    let mut support_vectors: Vec<Vec<f32>> = Vec::new();
    let mut support_labels: Vec<f32> = Vec::new();
    //let mut alphas:Vec<f32> = Vec::new();

    for i in 0..input_length {
        if alphas[i].abs() > 1e-3 {
            support_vectors.push(inputs[i].clone());
            support_labels.push(labels[i]);
            //alphas.push(alphas[i] as f32);
        }
    }

    model.support_vectors = support_vectors;
    model.support_labels = support_labels;
    model.alphas = alphas;

    //println!("Alphas saved, {:?}", model.support_labels);
}

pub fn mse_svm(expected: &Vec<f32>, prediction: &Vec<f32>) -> f32 {
    let error: f32 = expected.iter().zip(prediction).map(|(exp, pred)| (exp - pred).powi(2)).sum::<f32>();
    let result: f32 = error / expected.len() as f32;

    result
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn predict_svm(model_pointer: *mut SVMModel, inputs_pointer: *mut c_float) -> c_float {
    let model: &mut SVMModel = unsafe { &mut *model_pointer };
    let inputs_slice: &[f32] = unsafe { std::slice::from_raw_parts(inputs_pointer, model.dimensions as usize) };
    let inputs:Vec<f32> = inputs_slice.to_vec();

    //let result: f32 = inputs.iter().zip(&model.weight).map(|(i, w)| i * w).sum::<f32>() + model.biais;

    let result: f32 = model.support_vectors.iter()
        .zip(&model.alphas)
        .zip(&model.support_labels)
        .map(|((sv, alpha), label)| {
            *alpha * *label * get_kernel(model, &inputs, sv)
        })
        .sum::<f32>() + model.biais;

    // println!("Input: {:?}, Raw score: {}", inputs, result);
    result
}

#[no_mangle]
#[allow(dead_code)]
pub extern "C" fn free_svm(model_pointer: *mut SVMModel) {
    if !model_pointer.is_null() {
        unsafe {
            let _ = Box::from_raw(model_pointer);
        }
    }
}

pub extern "C" fn get_svm_state(model: *mut SVMModel) -> *mut c_char {
    let model = unsafe { &*model };
    let state = format!("SVs: {}, bias: {}", model.support_vectors.len(), model.biais);
    CString::new(state).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn save_svm(
    model: *const SVMModel,
    filepath: *const c_char,
) -> bool {
    let path_str = match unsafe { CStr::from_ptr(filepath) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid filepath");
            return false;
        }
    };

    let model_ref = unsafe { &*model };

    let json_str = match serde_json::to_string_pretty(model_ref) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("JSON serialization error: {}", e);
            return false;
        }
    };

    match File::create(path_str) {
        Ok(mut file) => {
            if let Err(e) = write!(file, "{}", json_str) {
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

#[no_mangle]
pub extern "C" fn svm_to_json(model: *const SVMModel) -> *mut c_char {
    let model_ref = unsafe { &*model };

    let json_obj = json!({
        "dim":model_ref.dimensions,
        "w":model_ref.weight,
        "b":model_ref.biais,
        "k":model_ref.kernel,
        "kv":model_ref.kernel_value,
        "sv":model_ref.support_vectors,
        "sl":model_ref.support_labels,
        "a":model_ref.alphas,
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
pub extern "C" fn loads_svm_model(filepath: *const c_char) -> *mut SVMModel {
    let path_str = match unsafe { CStr::from_ptr(filepath) }.to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Invalid filepath");
            return std::ptr::null_mut();
        }
    };

    let mut file = match File::open(path_str) {
        Ok(file) => file,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            return std::ptr::null_mut();
        }
    };

    let mut json_str = String::new();
    if let Err(e) = file.read_to_string(&mut json_str) {
        eprintln!("Error reading file: {}", e);
        return std::ptr::null_mut();
    }

    let model: SVMModel = match serde_json::from_str(&json_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error deserializing JSON: {}", e);
            return std::ptr::null_mut();
        }
    };

    Box::into_raw(Box::new(model))
}