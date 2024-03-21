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

#[allow(dead_code)]
#[allow(unused_variables)]
// use std::ops::Mul;
// use std::ops::Add;
// use std::default::Default;
// use std::convert::TryInto;

pub trait GetWeights
{
    fn get_weights(&self) -> Vec<f32>;
}
impl GetWeights for SingleLayerPerceptron
{
    fn get_weights(&self) -> Vec<f32>
    {
        self.weights.clone()
    }
}

pub trait SetWeights
{
    fn set_weights(&mut self, weights: Vec<f32>);
}
impl SetWeights for SingleLayerPerceptron
{
    fn set_weights(&mut self, weights: Vec<f32>)
    {
        self.weights = weights;
    }
}

// afficher les poids du models
pub trait ToString
{
    fn to_string(&self) -> String;
}
impl ToString for SingleLayerPerceptron
{
    fn to_string(&self) -> String
    {
        let mut res = "".to_string();
        for w in &self.weights {
            res.push_str(&w.to_string());
            res.push_str(", ");
        }
        res
    }
}

//  faire des prédictions
pub trait Predict
{
    fn guess(&self, inputs: &[f32]) -> f32;
    // fn predict(&self, inputs: &[T]);
}
impl Predict for SingleLayerPerceptron
{
    fn guess(&self, inputs: &[f32]) -> f32
    {
        let mut sum: f32 = 0.0;
        for (p,i) in self.weights.iter().zip(inputs) {
            sum = sum + *i * *p;
        }
        sum
    }
}


// Entraîner le modèle sur un ensemble de données d'entraînement
pub trait Fit
{
    fn fit(&mut self, datasets: (&Vec<Vec<f32>>, &Vec<f32>), learning_rate: f32);
}
impl Fit for SingleLayerPerceptron
{
    fn fit(&mut self, datasets: (&Vec<Vec<f32>>, &Vec<f32>), learning_rate: f32) {
        let (labels, features) = datasets;
        self.weights = vec![0.0; labels[0].len()];
        
        let mut time = 0;
        for (label, desired_output) in labels.iter().zip(features) {
            let predicted_output = self.guess(&label);
            let error = (desired_output.abs() - predicted_output.abs()).abs();
            let mut weights = vec![];
            for (weight, input) in self.weights.iter().zip(label) {
                let new_weight = &(weight + learning_rate * (desired_output - predicted_output) * input);
                weights.push(*new_weight);
            }
            self.set_weights(weights);
            self.history.push((time, error));
            time = time + 1;
        }
    }
}

pub struct SingleLayerPerceptron
{
    weights: Vec<f32>,
    history: Vec<(u32,f32)>
}

fn main()
{
    let mut sp = SingleLayerPerceptron {
        weights: vec![],
        history: vec![(0, 0.0)]
    };
    let celsius_temperatures = vec![
    vec![1.0, -50.0],
    vec![1.0, -40.0],
    vec![1.0, -30.0],
    vec![1.0, -20.0],
    vec![1.0, -10.0],
    vec![1.0, 0.0],
    vec![1.0, 10.0],
    vec![1.0, 20.0],
    vec![1.0, 30.0],
    vec![1.0, 40.0],
    vec![1.0, 50.0],
    vec![1.0, 60.0],
    vec![1.0, 70.0],
    vec![1.0, 80.0],
    vec![1.0, 90.0],
    vec![1.0, 100.0],
    vec![1.0, 110.0],
    vec![1.0, 120.0],
    vec![1.0, 130.0],
    vec![1.0, 140.0],
    vec![1.0, 150.0],
    vec![1.0, 160.0],
    vec![1.0, 170.0],
    vec![1.0, 180.0],
    vec![1.0, 190.0],
    vec![1.0, 200.0],
    vec![1.0, 210.0],
    vec![1.0, 220.0],
    vec![1.0, 230.0],
    vec![1.0, 240.0],
    vec![1.0, 250.0],
    vec![1.0, 260.0],
    vec![1.0, 270.0],
    vec![1.0, 280.0],
    vec![1.0, 290.0],
    vec![1.0, 300.0],
    ];

    let fahrenheit_temperatures = vec![
        -58.0, -40.0, -22.0, -4.0, 14.0, 32.0, 50.0, 68.0, 86.0, 104.0, 122.0, 140.0, 158.0, 176.0, 194.0, 212.0,
        230.0, 248.0, 266.0, 284.0, 302.0, 320.0, 338.0, 356.0, 374.0, 392.0, 410.0, 428.0, 446.0, 464.0, 482.0,
        500.0, 518.0, 536.0, 554.0, 572.0, 590.0,
    ];
    
    let datasets = (&celsius_temperatures, &fahrenheit_temperatures);

    sp.fit(datasets, 0.000001);
    println!( "{}", sp.to_string());

}


// impl SingleLayerPerceptron {

// }


// pub trait Save {
// // sauvegarder le modèle
//     fn save(&self) -> Sring;
// }

// pub trait Load {
// // charger le model
//     fn load(&self, model_path: str);
// }

// pub trait GetWeights {
// // recuperer les paremetres
//     fn get_weights<T>(&self) -> [T];
// }

// pub trait SetWeights {
// // definir/changer les parametres
//     fn set_weights(&self, weights: [T]);
// }


// impl GetWeights for SingleLayerPerceptron {
//     fn get_weights(&self) {
//         [1,2]
//     }
// }
    