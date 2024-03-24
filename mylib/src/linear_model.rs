/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   linear_model.rs                                           :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 19:38:54 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

// use std::ffi::{c_char, CStr};
// use std::fs::File;
// use std::io::prelude::*;

pub struct LinearModel
{
    weights: Vec<f32>,
    history: Vec<f32>,
}

impl LinearModel {
    pub fn new() -> Self {
        LinearModel {
            weights: vec![],
            history: vec![],
        }
    }

    fn get_weights(&self) -> Vec<f32>
    {
        self.weights.clone()
    }
    
    fn set_weights(&mut self, weights: Vec<f32>)
    {
        self.weights = weights;
    }

    pub fn guess(&self, inputs: Vec<f32>) -> f32
    {
        let mut sum: f32 = 0.0;
        for (p,i) in self.weights.iter().skip(1).zip(inputs) {
            sum = sum + i * *p;
        }
        sum = sum + self.weights[0];
        sum
    }

    #[no_mangle]
    pub extern "C" fn fit(&mut self, labels: *const f32, label_row: usize, label_col: usize, features: *const f32, feature_size: usize, learning_rate: f32, epochs: u32) {
        let labels = unsafe {std::slice::from_raw_parts(labels, (label_col * label_row) as usize)};
        let features = unsafe {std::slice::from_raw_parts(features, feature_size as usize)};
    
        self.weights = vec![0.0; label_row + 1];
        let mut epochs = epochs;
        let mut time = 0;
        let data_size = if label_row < feature_size {label_row} else {feature_size};

        while epochs != 0 {
            let mut count_label = 0;
            for count_feature in 0..data_size {
                let desired_output = features[count_feature];

                let mut input: Vec<f32> = vec![];
                for j in 0..label_col {
                    let elem = labels[count_label + j];
                    input.push(elem);
                }
                count_label  = count_label + label_col;
                let predicted_output = self.guess(input.clone());
                let error = (desired_output - predicted_output).abs();
                let mut weights: Vec<f32> = vec![0.0];
                for j in 0..label_col {
                    weights.push(self.weights[j+1] + learning_rate * (desired_output - predicted_output) * input[j]);
                }
                weights[0] = self.weights[0] + learning_rate * (desired_output - predicted_output);
                self.set_weights(weights);
                self.history.push(error);
                time = time + 1;
            }
            epochs = epochs - 1;
        }
    }

    #[no_mangle]
    pub extern "C" fn to_string(&self) -> *const u8 {
        let weights = self.get_weights();
        let mut res = "".to_string();
        for w in weights {
            res.push_str(&w.to_string());
            res.push_str(",");
        }
        let leaked_res = String::leak(res);
        leaked_res.as_ptr()
    }

    #[no_mangle]
    pub extern "C" fn get_history(&self) -> *const f32
    {
        let leaked_arr = Vec::leak(self.history.clone());
        leaked_arr.as_ptr()
    }
    
}
