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
use std::ops::Mul;
use std::ops::Add;
use std::default::Default;
use std::convert::TryInto;

pub trait GetWeights<T>
{
    fn get_weights(&self) -> Vec<T>;
}
impl<T, U> GetWeights<T> for SingleLayerPerceptron<T, U>
where T: Clone 
{
    fn get_weights(&self) -> Vec<T>
    {
        self.weights.clone()
    }
}

pub trait SetWeights<T>
{
    fn set_weights(&mut self, weights: Vec<T>);
}
impl<T, U> SetWeights<T> for SingleLayerPerceptron<T, U>
{
    fn set_weights(&mut self, weights: Vec<T>)
    {
        self.weights = weights;
    }
}

// afficher les poids du models
pub trait ToString<T>
{
    fn to_string(&self) -> String;
}
impl<T, U> ToString<T> for SingleLayerPerceptron<T, U>
where T : std::fmt::Display
{
    fn to_string(&self) -> String
    {
        let mut res = "".to_string();
        for w in &self.weights {
            res.push_str(&w.to_string());
            res.push_str(" ");
        }
        res
    }
}

//  faire des prédictions
pub trait Predict<T, U>
{
    fn predict(&self, inputs: &[T]) -> U;
    fn guess(&self, inputs: &[T]) -> T;
}
impl<T> Predict<T, T> for SingleLayerPerceptron<T, T>
where T: Add<Output=T> + Mul<Output=T> + Default + Copy
{
    fn guess(&self, inputs: &[T]) -> T
    {
        let mut sum: T = Default::default();
        for (p,i) in self.weights.iter().zip(inputs) {
            sum = sum + *i * *p;
        }
        sum
    }

    fn predict(&self, inputs: &[T]) -> T
    {
        let sum = self.guess(inputs);
        sum
    }
}


// Entraîner le modèle sur un ensemble de données d'entraînement
pub trait Fit<T>
{
    fn fit(&mut self, datasets: (&Vec<Vec<T>>, &Vec<T>), learning_rate: f32);
}
impl<T, U> Fit<T> for SingleLayerPerceptron<T, U> 
where T: Mul<T, Output=T> + Default + Clone,
f32: Mul<T> + Add<T>
{
    fn fit(&mut self, datasets: (&Vec<Vec<T>>, &Vec<T>), learning_rate: f32) {
        let (labels, features) = datasets;
        self.weights = vec![< T as Default>::default(); labels[0].len().try_into().unwrap()];

        for (label, feature) in labels.iter().zip(features) {
            let res = self.guess(&label);
            for i in 0..self.weights.len() {
                self.weights[i] = self.weights[i] + learning_rate * (feature - res);
            }
        }
    }
}

pub struct SingleLayerPerceptron<T, U>
{
    weights: Vec<T>,
    history: (Vec<T>, U)
}

fn main()
{
    let mut sp = SingleLayerPerceptron {
        weights: vec![],
        history: (vec![], 0)
    };
    
    let labels = vec![vec![1, 2, 3], vec![5, 2, 3], vec![1, 2, 3]];
    let features = vec![1, 2, 1];
    let datasets = (&labels, &features);
    
    sp.fit(datasets, 0.3);
    
    println!( "{}", sp.to_string());
    sp.set_weights(vec![3,2,1]);
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
    