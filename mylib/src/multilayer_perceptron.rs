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