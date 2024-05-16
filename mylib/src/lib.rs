/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   lib.rs                                                    :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/18 21:39:57 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/18 21:39:57 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */


pub mod linear_model;
pub mod multilayer_perceptron;
pub mod radical_basis_function_network;
pub mod support_vector_machine;
pub mod utils;

#[allow(unused_imports)]
pub use multilayer_perceptron::{MultiLayerPerceptron , init_mlp, train_mlp, predict_mlp, free_mlp, save_mlp_model};

#[allow(unused_imports)]
pub use linear_model::{LinearModel, init_linear_model, train_linear_model, predict_linear_model,free_linear_model, save_linear_model, load_linear_model};
