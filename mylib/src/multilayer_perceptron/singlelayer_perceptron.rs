/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   singlelayer_perceptron.rs                                 :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/19 23:27:37 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/19 23:27:37 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */
#[allow(unused_variables)]
#[allow(dead_code)]
use std::fs::File;
use std::io::prelude::*;

pub struct SingleLayerPerceptron
{
    weights: Vec<f32>,
    history: Vec<f32>
}

impl SingleLayerPerceptron {
    pub fn new() -> Self {
        SingleLayerPerceptron {
            weights: vec![],
            history: vec![]
        }
    }
}

//  faire des prédictions
pub trait Predict
{
    extern "C" fn guess(&self, inputs: &[f32]) -> f32;
    // fn predict(&self, inputs: &[T]);
}
impl Predict for SingleLayerPerceptron
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
}

// Entraîner le modèle sur un ensemble de données d'entraînement
pub trait Fit
{
    extern "C" fn fit(&mut self, datasets: (&Vec<Vec<f32>>, &Vec<f32>), learning_rate: f32, epochs: u32, tolerance: f32);
}
impl Fit for SingleLayerPerceptron
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

pub trait GetWeights
{
    extern "C" fn get_weights(&self) -> Vec<f32>;
}
impl GetWeights for SingleLayerPerceptron
{
    #[no_mangle]
    extern "C" fn get_weights(&self) -> Vec<f32>
    {
        self.weights.clone()
    }
}

pub trait SetWeights
{
    extern "C" fn set_weights(&mut self, weights: Vec<f32>);
}
impl SetWeights for SingleLayerPerceptron
{
    #[no_mangle]
    extern "C" fn set_weights(&mut self, weights: Vec<f32>)
    {
        self.weights = weights;
    }
}

// afficher les poids du models
pub trait ToString
{
    extern "C" fn to_string(&self) -> String;
}
impl ToString for SingleLayerPerceptron
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

pub trait GetHistory
{
    extern "C" fn get_history(&self) -> Vec<f32>;
}
impl GetHistory for SingleLayerPerceptron
{
    #[no_mangle]
    extern "C" fn get_history(&self) -> Vec<f32>
    {
        self.history.clone()
    }
}

pub trait Save {
// sauvegarder le modèle
    
    extern "C" fn save(&self, filepath: &str) ->std::io::Result<()>;
}
impl Save for SingleLayerPerceptron
{
    #[no_mangle]
    extern "C" fn save(&self, filepath: &str) -> std::io::Result<()>
    {
        let mut file = File::create(filepath)?;
        file.write_all(self.to_string().as_bytes())?;
        Ok(())
    }
}

pub trait Load {
// charger le model
    extern "C" fn load(&mut self, filepath: &str);
}
impl Load for SingleLayerPerceptron
{
    #[no_mangle]
    extern "C" fn load(&mut self, filepath: &str)
    {
    }
}

fn main()
{
    let mut sp = SingleLayerPerceptron::new();
    let celsius_temperatures = vec![
        vec![0.0], vec![11.0], vec![27.0], vec![6.0],
        vec![13.0], vec![39.0], vec![19.0], vec![25.0],
        vec![30.0], vec![-15.0], vec![14.0], vec![33.0],
        vec![-5.0], vec![20.0], vec![22.0], vec![-10.0],
        vec![17.0], vec![37.0], vec![24.0], vec![-20.0],
        vec![8.0], vec![35.0], vec![-30.0], vec![21.0],
        vec![10.0], vec![-25.0], vec![-2.0], vec![-40.0],
        vec![16.0], vec![29.0], vec![7.0], vec![-35.0],
        vec![36.0], vec![-22.0], vec![-38.0], vec![28.0],
        vec![-45.0], vec![38.0], vec![3.0], vec![18.0],
        vec![-12.0], vec![-18.0], vec![32.0], vec![-8.0],
        vec![15.0], vec![2.0], vec![-33.0], vec![26.0],
        vec![-50.0], vec![12.0], vec![-3.0], vec![5.0],
    ];
    
    let fahrenheit_temperatures = vec![
        32.000, 51.800, 80.600, 42.800, 55.400, 102.200, 66.20, 77.0, 87.8,
        5.0, 57.2, 91.4, 23.0, 68.0, 71.6, 14.0, 62.6, 98.6, 75.2, -4.0,
        46.4, 95.0, -22.0, 69.8, 50.0, -13.0, 28.4, -40.0, 60.8, 84.2, 
        44.6, 84.2, 102.2, 37.4, 64.4, 105.8, 39.2, 14.0, 17.6, 89.6,
        46.4, 30.2, -22.0, 27.0, 21.2, -27.4, -25.6, 73.4, -9.4, -40.0,
        -8.0, 91.4, -26.0, -37.4, -16.6, -54.2,
    ];
    
    
    let datasets = (&celsius_temperatures, &fahrenheit_temperatures);
    sp.fit(datasets, 0.0001, 5, 0.5);
    // println!("5C = {}F", sp.guess(&[1.0, 5.0]));
    println!("{}", sp.to_string());

    // let res = sp.save("../models/firstmodel.txt");
}
