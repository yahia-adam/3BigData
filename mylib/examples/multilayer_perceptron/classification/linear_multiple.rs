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

// #[allow(unused_imports)]
// use mylib::{free_mlp, init_mlp, predict_mlp, save_mlp_model, train_mlp, MultiLayerPerceptron};

// use std::{
//     env,
//     ffi::{c_char, CString},
//     vec,
// };

// fn main() {
//     let model_path: Option<String> = env::args().nth(1);
//     if let Some(path) = model_path {
//         // file to save models
//         let filepath_cstr: CString = CString::new(path).expect("Failed to create CString");
//         let filepath_ptr: *const c_char = filepath_cstr.as_ptr();

//         let sample_inputs: Vec<f32> = vec![0.5, 0.4, 1.5, 1.3, 2.5, 2.1, 3.5, 3.2, 4.5, 4.0, 0.5, 0.6, 1.5, 1.7, 2.5, 2.9, 3.5, 3.8, 4.5, 4.6];
//         let sample_outputs: Vec<f32> = vec![-1.0, -1.0, -1.0, -1.0, -1.0, 1.0,  1.0,  1.0,  1.0,  1.0];

//         let row: usize = 4;
//         let input_col: usize = 2;
//         let output_col: usize = 1;
//         let inputs: *mut f32 = Vec::leak(sample_inputs).as_mut_ptr();
//         let outputs: *mut f32 = Vec::leak(sample_outputs).as_mut_ptr();

//         // model init
//         let npl: *mut usize = Vec::leak(vec![2, 1]).as_mut_ptr();
//         let model: *mut MultiLayerPerceptron = init_mlp(npl, 2);

//         save_mlp_model(model, filepath_ptr);
//         // train model
//         train_mlp(
//             model, inputs, outputs, input_col, output_col, row, 0.1, 20000, true,
//         );
//         // save_mlp_model(model, filepath_ptr2);
        
//         println!("\n<----------------- test ---------------------->");
//         let tests: Vec<Vec<f32>> = vec![
//             vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 3.0], vec![3.0, 5.0], vec![4.0, 7.0],
//             vec![0.0, 2.0], vec![1.0, 4.0], vec![2.0, 6.0], vec![3.0, 8.0], vec![4.0, 10.]
//         ];
//         let labels: Vec<f32> = vec![
//             -1.0, -1.0, -1.0, -1.0, -1.0,
//              1.0,  1.0,  1.0,  1.0,  1.0
//         ];

//         for i in 0..tests.len() {
//             let test_ptr: *mut f32 = Vec::leak(tests[i].clone()).as_mut_ptr();
//             let res_ptr: *mut f32 = predict_mlp(model, test_ptr, tests[i].len(), true);
//             let result: Vec<f32> = unsafe { Vec::from_raw_parts(res_ptr, 1, 1) };
//             println!("valeur attendu : {:?}: valeur predit :{:?} ", labels[i], result);
//         }
//         println!("");
//         free_mlp(model);
//     }
// }
