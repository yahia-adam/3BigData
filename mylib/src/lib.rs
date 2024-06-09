/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   lib.rs                                                    :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */


pub mod linear_model;
pub mod multilayer_perceptron;
pub mod radical_basis_function_network;
pub mod support_vector_machine;

#[allow(unused_imports)]
pub use multilayer_perceptron::{MultiLayerPerceptron, init_mlp, train_mlp, predict_mlp, free_mlp, save_mlp_model};

#[allow(unused_imports)]
pub use linear_model::{LinearModel, init_linear_model, train_linear_model, predict_linear_model,free_linear_model, save_linear_model, load_linear_model};

#[allow(unused_imports)]
pub use radical_basis_function_network::{RadicalBasisFunctionNetwork, init_rbf,
                                         train_rbf_regression, train_rbf_rosenblatt,
                                         predict_rbf_regression, predict_rbf_classification,
                                         free_rbf, save_rbf_model, rbf_to_json};