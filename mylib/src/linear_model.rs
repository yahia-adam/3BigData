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

use std::fs::File;
use std::io::prelude::*;
use crate::utils::model::{Predict, Fit, GetWeights, SetWeights, ToString, Save, Load, GetHistory};

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
}

impl Predict for LinearModel
{
    #[no_mangle]
    extern "C" fn guess(&self, inputs: &[f32]) -> f32
    {
        let mut sum: f32 = 0.0;
        for (p,i) in self.weights.iter().skip(1).zip(inputs) {
            sum = sum + *i * *p;
        }
        sum = sum + self.weights[0];
        sum
    }
    #[no_mangle]
    extern "C" fn predict(&self, inputs: &[f32]) {

    }
}

impl Fit for LinearModel
{
    #[no_mangle]
    extern "C" fn fit(&mut self, datasets: (&Vec<Vec<f32>>, &Vec<f32>), learning_rate: f32, epochs: u32, _tolerance: f32) {
        let (labels, features) = datasets;
        self.weights = vec![0.0; labels[0].len() + 1];
        let mut epochs = epochs;
        let mut time = 0;
        while epochs != 0 {
            for (label, desired_output) in labels.iter().zip(features) {
                let predicted_output = self.guess(&label);
                let error = (desired_output - predicted_output).abs();
                for i in 0..label.len() {
                    self.weights[i+1] = self.weights[i] + learning_rate * (desired_output - predicted_output) * label[i];
                }
                self.weights[0] = self.weights[0] + learning_rate * (desired_output - predicted_output);
                println!("{}", self.to_string());
                self.history.push(error);
                time = time + 1;
            }
            epochs = epochs - 1;
        }
    }
}

impl GetWeights for LinearModel
{
    #[no_mangle]
    extern "C" fn get_weights(&self) -> Vec<f32>
    {
        self.weights.clone()
    }
}

impl SetWeights for LinearModel
{
    #[no_mangle]
    extern "C" fn set_weights(&mut self, weights: Vec<f32>)
    {
        self.weights = weights;
    }
}

impl ToString for LinearModel
{
    #[no_mangle]
    extern "C" fn to_string(&self) -> String
    {
        let mut res = "".to_string();
        for w in &self.weights {
            res.push_str(&w.to_string());
            res.push_str(",");
        }
        res
    }
}

impl Save for LinearModel
{
    #[no_mangle]
    extern "C" fn save(&self, filepath: &str) -> std::io::Result<()>
    {
        let mut file = File::create(filepath)?;
        file.write_all(self.to_string().as_bytes())?;
        Ok(())
    }
}

impl Load for LinearModel
{
    #[no_mangle]
    extern "C" fn load(&mut self, filepath: &str)
    {
    }
}

impl GetHistory for LinearModel
{
    #[no_mangle]
    extern "C" fn get_history(&self) -> Vec<f32>
    {
        self.history.clone()
    }
}
