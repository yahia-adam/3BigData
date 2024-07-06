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
use ndarray::prelude::*;
use ndarray_linalg::*;
use serde_json::{self};
use std::ffi::CStr;
use ndarray_rand::{rand};
use ndarray_rand::rand::Rng;
use std::slice::{from_raw_parts};
use libm::*;
use std::fs::File;
use std::io::{Write, BufReader};
use std::os::raw::c_char;
use pbr::ProgressBar;

#[derive(Serialize, Deserialize, Debug)]
pub struct RadicalBasisFunctionNetwork {
    weights : Vec<f32>,
    centers : Vec<Vec<f32>>,
    gamma : f32,
    pub train_loss: Vec<f32>,
    pub test_loss: Vec<f32>,
    pub train_accuracy: Vec<f32>,
    pub test_accuracy: Vec<f32>
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
        train_loss: vec![],
        test_loss: vec![],
        train_accuracy: vec![],
        test_accuracy: vec![],
    };

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

pub fn euclid(x : &[f32], y : &[f32]) -> f32{
    let mut res = 0f32;
    for i in 0..(x.len()){
        res += powf(y[i] - x[i], 2f32);
    }
    sqrtf(res)
}

pub fn get_rand_centers(data : &[f32], cluster_num : i32, sample_count : i32, inputs_size : i32)-> Vec<Vec<f32>>{
    let mut centers = Vec::with_capacity(cluster_num as usize);
    let mut copy_data_size = sample_count;
    let mut copy_data = data.to_vec();

    for _ in 0..cluster_num{
        let initial_center = rand::thread_rng().gen_range(0..copy_data_size);
        let data_initial_center = &copy_data[(initial_center * inputs_size) as usize..((initial_center + 1) * inputs_size) as usize];
        centers.push(data_initial_center.to_vec());
        copy_data_size -= 1;
        copy_data.remove(initial_center as usize);
    }
    centers
}

pub fn mean(cluster : &[&[f32]], inputs_size : i32)-> Vec<f32>{
    let mut average = Vec::with_capacity(inputs_size as usize);
    for dimension in 0..inputs_size as usize{
        average.push(0f32);
        for points in 0..cluster.len(){
            average[dimension] += cluster[points][dimension];
        }
        average[dimension] /= cluster.len() as f32;
    }
    average
}

pub fn lloyd(data : &[f32], cluster_num : i32, iterations : i32, sample_count : i32, inputs_size : i32)-> Vec<f32>{
    if cluster_num == sample_count {
        return data.to_vec();
    }
    let mut clusters = Vec::with_capacity(cluster_num as usize);
    for _ in 0..cluster_num{
        clusters.push(Vec::new());
    }
    let mut sites = get_rand_centers(data, cluster_num, sample_count, inputs_size).to_vec();
    for _ in 0..iterations as usize{
        for point in 0..sample_count as usize{
            let data_point = &data[(point * inputs_size as usize)..((point + 1) * inputs_size as usize)];
            let mut closest_site_number = 0usize;
            let mut closest_site_distance = euclid(sites[closest_site_number].as_slice(), data_point);
            for site_number in 1..sites.len(){
                let site_distance = euclid(sites[site_number].as_slice(), data_point);
                if site_distance < closest_site_distance {
                    closest_site_number = site_number;
                    closest_site_distance = site_distance;
                }
            }
            clusters[closest_site_number].push(data_point);
        }
        for m in 0..cluster_num as usize{
            sites[m] = mean(clusters[m].as_slice(), inputs_size);
            clusters[m].clear();
        }
    }
    let mut sites_flat = Vec::with_capacity((cluster_num * inputs_size) as usize);
    for i in 0..cluster_num as usize{
        for j in 0..inputs_size as usize{
            sites_flat.push(sites[i][j]);
        }
    }
    sites_flat
}

#[no_mangle]
pub extern "C" fn train_rbf_regression(model: *mut RadicalBasisFunctionNetwork, sample_inputs_flat: *mut f32, expected_outputs: *mut f32, inputs_size: i32, sample_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let cluster_num = model.weights.len() as i32;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, (inputs_size * sample_count) as usize)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count as usize)
    };
    let cluster_points = lloyd(sample_inputs_flat, cluster_num, 10, sample_count, inputs_size);

    let mut pb = ProgressBar::new(sample_count as u64);
    pb.show_tick = false;
    pb.show_speed = false;
    pb.show_percent = false;
    pb.show_counter = false;

    let mut phi = Array::default((sample_count as usize, cluster_num as usize));

    for i in 0..sample_count as usize {
        let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
        for j in 0..cluster_num as usize {
            let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1) * inputs_size as usize)];
            phi[(i, j)] = expf(-model.gamma * euclid(xi, cluster_pointsj) * euclid(xi, cluster_pointsj));
            for n in 0..inputs_size as usize {
                model.centers[j][n] = cluster_pointsj[n];
            }
        }
    }

    let y = Array::from(expected_outputs.to_vec());
    let phitphi = phi.t().dot(&phi);
    let phitphi_inv = phitphi.inv().unwrap();
    let w = (phitphi_inv.dot(&phi.t())).dot(&y);

    for i in 0..cluster_num as usize {
        model.weights[i] = w[i];
    }

    let mut y_pred = vec![0.0; sample_count as usize];
    for i in 0..sample_count as usize {
        let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
        y_pred[i] = predict_rbf_regression_slice(model, xi);
    }
    let final_loss = mse_epoch(expected_outputs, &y_pred);
    model.train_loss.push(final_loss);

    pb.finish_println(&format!("Training completed - Final loss: {:.4}\n", final_loss));
}

fn predict_rbf_regression_slice(model : &RadicalBasisFunctionNetwork, inputs : &[f32])-> f32{
    let mut res = 0f32;
    for i in 0..model.weights.len(){
        res += model.weights[i] * expf(-model.gamma * euclid(inputs, model.centers[i].as_slice()) * euclid(inputs, model.centers[i].as_slice()));
    }
    res
}

fn predict_rbf_classification_slice(model : &RadicalBasisFunctionNetwork, inputs : &[f32])-> f32{
    let pred = predict_rbf_regression_slice(model, inputs);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

#[no_mangle]
pub extern "C" fn predict_rbf_regression(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32) -> f32{
    let model = unsafe {
      model.as_mut().unwrap()
    };

    let inputs = unsafe{
      from_raw_parts(inputs, model.centers[0].len())
    };

    predict_rbf_regression_slice(model, inputs)

}

fn mse_epoch(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let n = y_true.len() as f32;
    let sum_squared_error: f32 = y_true.iter().zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum();
    sum_squared_error / n
}

fn accuracy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let correct_predictions: usize = y_true.iter().zip(y_pred.iter())
        .filter(|(&true_val, &pred_val)| {
            (true_val >= 0.5 && pred_val >= 0.5) || (true_val < 0.5 && pred_val < 0.5)
        })
        .count();
    correct_predictions as f32 / y_true.len() as f32
}

#[no_mangle]
pub extern "C" fn train_rbf_rosenblatt(model: *mut RadicalBasisFunctionNetwork, sample_inputs_flat: *mut f32, expected_outputs: *mut f32, iterations_count: i32, alpha: f32, inputs_size: i32, sample_count: i32) {
    let model = unsafe {
        model.as_mut().unwrap()
    };
    let cluster_num = model.weights.len() as i32;
    let sample_inputs_flat = unsafe {
        from_raw_parts(sample_inputs_flat, (inputs_size * sample_count) as usize)
    };
    let expected_outputs = unsafe {
        from_raw_parts(expected_outputs, sample_count as usize)
    };
    let cluster_points = lloyd(sample_inputs_flat, cluster_num, 10, sample_count, inputs_size);

    for j in 0..cluster_num as usize {
        let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1) * inputs_size as usize)];
        model.centers[j] = cluster_pointsj.to_vec();
    }

    for epoch in 0..iterations_count as usize {
        let mut pb = ProgressBar::new(sample_count as u64);
        pb.format("[=>-]");
        pb.message(format!("Epoch {}/{} - loss: {:.4} - accuracy: {:.2} ", epoch + 1, iterations_count, 0.0, 0.0).as_str());
        pb.show_tick = true;
        pb.show_speed = false;
        pb.show_percent = false;
        pb.show_counter = false;

        let mut y_true: Vec<f32> = vec![];
        let mut y_pred: Vec<f32> = vec![];

        for k in 0..sample_count as usize {
            let x = &sample_inputs_flat[(k * inputs_size as usize)..((k + 1) * inputs_size as usize)];
            let yk = expected_outputs[k];
            let gk = predict_rbf_classification_slice(model, x);

            y_true.push(yk);
            y_pred.push(gk);

            for i in 0..cluster_num as usize {
                let rbf_value = expf(-model.gamma * euclid(x, &model.centers[i]).powi(2));
                model.weights[i] += alpha * (yk - gk) * rbf_value;
            }

            let current_loss = mse_epoch(&y_true, &y_pred);
            let current_accuracy = accuracy(&y_true, &y_pred);
            pb.message(format!("Epoch {}/{} - loss: {:.4} - accuracy: {:.2} ", epoch + 1, iterations_count, current_loss, current_accuracy).as_str());
            pb.inc();
        }

        let epoch_loss = mse_epoch(&y_true, &y_pred);
        let epoch_accuracy = accuracy(&y_true, &y_pred);
        model.train_loss.push(epoch_loss);
        model.train_accuracy.push(epoch_accuracy);

        pb.finish_println(&format!(
            "Epoch {}/{} - loss: {:.4} - accuracy: {:.2} ",
            epoch + 1, iterations_count, epoch_loss, epoch_accuracy
        ));
    }
}

#[no_mangle]
pub extern "C" fn predict_rbf_classification(model : *mut RadicalBasisFunctionNetwork, inputs : *mut f32)-> f32{
    let pred = predict_rbf_regression(model, inputs);
    return if pred >= 0.0 { 1.0 } else { -1.0 };
}

#[no_mangle]
pub extern "C" fn rbf_to_json(path : *const c_char)-> *mut RadicalBasisFunctionNetwork{
    let path = unsafe{
        CStr::from_ptr(path).to_str().unwrap()
    };
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let model = serde_json::from_reader(reader).unwrap();

    let boxed_model = Box::new(model);
    let pointer = Box::leak(boxed_model);
    pointer
}

#[no_mangle]
pub extern "C" fn free_rbf(model : *mut RadicalBasisFunctionNetwork){
    unsafe{
        let _ = Box::from_raw(model);
    }
}

#[no_mangle]
pub extern "C" fn save_rbf_model(model : *mut RadicalBasisFunctionNetwork, path : *const c_char){
    let model = unsafe{
        model.as_mut().unwrap()
    };

    let path = unsafe{
        CStr::from_ptr(path).to_str().unwrap()
    };

    let serialized = serde_json::to_string(&model).unwrap();
    let mut output = File::create(path).unwrap();
    write!(output, "{}", &serialized).unwrap();
}