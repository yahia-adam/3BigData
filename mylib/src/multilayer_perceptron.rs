/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   multilayer_perceptron.rs                                  :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 14:20:22 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 14:20:22 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

use rand::Rng;
use serde::{Deserialize, Serialize};

// derive attribute to add functionalities to the MLP structure
// serialize and deserialize indicates that the structure can be transformed in and from serialization formats
#[derive(Serialize, Deserialize, Debug)]
pub struct StructMLP{ // the following structure represents a multi-layer perceptron
    d: Vec<i32>, // vector of int represents the number of neurons in each layer or the number of layers
    w: Vec<Vec<Vec<f32>>>, // un vecteur tridimensionnel, represents the weights of the neurons
    x: Vec<Vec<f32>>, // un vecteur de vecteurs, represents the entries of the MLP
    delta: Vec<Vec<f32>> // un vecteur de vecteurs, represents errors / corrections to the MLP
}

// implementation of the StructMLP
impl StructMLP {
    // perform a forward pass in the neural network
    /**
     &mut self -> indicates that the forward pass method modify the instance of the StructMLP
     sample_input: &Vec<f32> -> network entries are passed by reference, to avoid copying the entire vector.
     is_classification: bool -> flag for whether the network should use tanh when passing forward

     the forward_pass method implements a forward propagation mechanism in an MLP
     updating the values of the neurons in each layer according to the inputs, weights, and activation
     functions (tanh in our case) if necessary
     **/
    pub extern "C" fn forward_pass(&mut self, sample_inputs: &Vec<f32>, is_classification: bool){
        // installation of the entries
        // initialize the first layer of network inputs with values of the sample_inputs vector
        // on commence avec 1 au cas ou le 0 est reserve pour le biais
        for j in 1..(self.d[0] + 1) as usize{
            self.x[0][j] = sample_inputs[j - 1];
        }

        // forward passage through layers
        // traverse each layer starting from the 1st layer to the last output layer
        for l in 1..(self.d.len()) as usize{
            // calculates the weighted sum of the previous layer's inputs with the weights
            for j in 1..(self.d[l] + 1) as usize{
                let mut sum_result = 0.0f32; // accumulates the product of the weights and inputs
                for i in 0..(self.d[l - 1] + 1) as usize{
                    sum_result += self.w[l][i][j] * self.x[l - 1][i];
                }

                // store the calculated in self.x[l][j]
                self.x[l][j] = sum_result;

                // applying activation functions
                // if the index of the current layer is lower than the last one (which means it is a hidden layer)
                // or if it is a classification problem -> we apply the tanh activation function at self.x[l][j]
                if(l < (self.d.len() - 1) as usize) || is_classification{
                    // the tanh function is an activation function that normalizes values between -1 and 1
                    self.x[l][j] = self.x[l][j].tanh();
                }
            }
        }
    }
}

impl StructMLP{
    // retro propagation with stochastic gradient descent  algorithm to train the MLP network
    /**
     the following method trains an MLP using a stochastic gradient descent approach with backpropagation
     it performs a forward pass to predict the outputs, calculates the errors/deltas and updates the weights
     using the derivative of the activation function and the learning rate. each iteration takes a sample.
     **/
    pub extern "C" fn train_stochastic_gradient_backpropagation(&mut self, flattened_data_inputs: &Vec<f32>, flattened_expected_outputs: &Vec<f32>, is_classification: bool, alpha: f32, iterations_counts: i32){
        let last = (self.d.len() - 1) as usize; // the last index of the layer of the network
        let input_dim = self.d[0] as usize; // the number of entries, the dimension of the entry layer
        let output_dim = self.d[last] as usize; // the number of exits, the dimension of the exit layer
        let samples_count = flattened_data_inputs.len() / input_dim as usize; // the number of samples in the input data

        // for loop executed with a certain number of iterations for the training
        for _it in 0..iterations_counts as usize{
            // each iteration chooses a random sample in the input data
            let k = rand::thread_rng().gen_range(0..samples_count) as usize;
            // the input values for the chosen sample, taken from flattened_data_inputs
            let sample_inputs = flattened_data_inputs[(k * input_dim)..((k + 1) * input_dim)].to_vec();
            // the corresponding expected output values, taken from flattened_expected_outputs
            let sample_expected_outputs = &flattened_expected_outputs[k * output_dim..(k + 1) * output_dim];

            // is called to obtain the network output values for the chosen sample
            self.forward_pass(&sample_inputs, is_classification);

            // pour tous les neurones j de la dernière couche last on calcule delta[last][j]
            for j in 1..(self.d[last] + 1) as usize{
                self.delta[last][j] = self.x[last][j] - sample_expected_outputs[j - 1];
                // if is_classification is true, a derivative of the activation function is applied to adjust the delta
                if is_classification{
                    self.delta[last][j] = (1.0f32 - self.x[last][j] * self.x[last][j]) * self.delta[last][j];
                }
            }

            // on en déduit pour tous les autres neurones de l'avant dernière couche à la première
            // backward pass
            for l in (1..last + 1).rev(){
                for i in 0..(self.d[l - 1] + 1) as usize{
                    let mut sum_result = 0.0f32;
                    for j in 1..(self.d[l] + 1) as usize{
                        sum_result += self.w[l][i][j] * self.delta[l][j];
                    }
                    // used to fit the deltas using the derivative of the activation function tanh
                    self.delta[l - 1][i] = (1.0f32 - self.x[l - 1][i] * self.x[l - 1][i]) * sum_result;
                }
            }

            // puis on met à jour tous les w[l][i][j] weights

            for l in 1..last + 1{
                for i in 0..(self.d[l - 1] + 1) as usize{
                    for j in 1..(self.d[l] + 1) as usize{
                        // we use the stochastic gradient formula (where alpha is the learning rate)
                        self.w[l][i][j] += -alpha * self.x[l - 1][i] * self.delta[l][j];
                    }
                }
            }

        }
    }
}

