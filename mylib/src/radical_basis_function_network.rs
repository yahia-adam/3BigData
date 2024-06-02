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


use std::slice::from_raw_parts;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value::Array;

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
        average[dimension] += cluster.len() as f32;
    }
    average
}

pub fn lloyd(data : &[f32], cluster_num : i32, iterations : i32, sample_count : i32, inputs_size : i32)-> Vec<f32>{
    if(cluster_num == sample_count){
        return data.to_vec();
    }
    let mut clusters = Vec::with_capacity(cluster_num as usize);
    for _ in 0..cluster_num{
        clusters.push(Vec::new());
    }
    let mut sites = get_rand_centers(data, cluster_num, sample_count, inputs_size).to_vec();
    for _ in 0..iterations as usize{
        for point in 0..sample_count as usize{
            let data_point = &data[(point * inputs_size as usize)..((point + 1)) * inputs_size as usize];
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
pub extern "C" fn train_rbf_regression(model : *mut RadicalBasisFunctionNetwork, sample_inputs_flat : *mut f32, expected_outputs : *mut f32, inputs_size : i32, sample_count : i32){
    let model = unsafe{
        model.as_mut().unwrap()
    };
    let cluster_num = model.weights.len() as i32;
    let sample_inputs_flat = unsafe{
        from_raw_parts(sample_inputs_flat, (inputs_size * sample_count) as usize)
    };
    let expected_outputs = unsafe {
      from_raw_parts(expected_outputs, sample_count as usize)
    };
    let cluster_points = lloyd(sample_inputs_flat, cluster_num, 10, sample_count, inputs_size);

    let mut phi = Array::default((sample_count as usize, cluster_num as usize));

    for i in 0..sample_count as usize{
        let xi = &sample_inputs_flat[(i * inputs_size as usize)..((i + 1) * inputs_size as usize)];
        for j in 0..cluster_num as usize{
            let cluster_pointsj = &cluster_points[(j * inputs_size as usize)..((j + 1 ) * inputs_size as usize)];
            phi[(i, j)] = (-model.gamma * euclid(xi, cluster_pointsj).powi(2)).exp();
            for n in 0..inputs_size as usize{
                model.centers[j][n] = cluster_pointsj[n];
            }
        }
    }

    let y = Array::from(expected_outputs.to_vec());
    let phitphi = phi.t().dot(&phi);
    let phitphi_inv = phitphi.inv().unwrap();
    let w = (phitphi_inv.dot(&phi.t())).dot(&y);

    for i in 0..cluster_num as usize{
        model.weights[i] = w[i];
    }
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

fn predict_rbf_regression_slice(model : &RadicalBasisFunctionNetwork, inputs : &[f32])-> f32{
    let mut res = 0f32;
    for i in 0..model.weights.len(){
        res += model.weights[i] * (-model.gamma * euclid(inputs, model.centers[i].as_slice()).powi(2)).exp();
    }
    res
}

