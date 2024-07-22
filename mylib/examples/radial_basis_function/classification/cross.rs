// #[allow(unused_imports)]
// use mylib::{
//     RadialBasisFunctionNetwork, init_rbf, train_rbf_regression, train_rbf_rosenblatt,
//     predict_rbf_regression, predict_rbf_classification, free_rbf, generate_dataset};

// fn main() {
//     let (mut x, mut y) = generate_dataset();
//     let data_size = y.len();
//     let input_dim = x[0].len();
//     let mut x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
//     let x_ptr: *mut f32 = x_flatten.as_mut_ptr();
//     let y_ptr: *mut f32 = y.as_mut_ptr();

//     let rbf_model: *mut RadialBasisFunctionNetwork = init_rbf(input_dim as i32, 32, 1.0);
//     train_rbf_rosenblatt(rbf_model, x_ptr, y_ptr, 9543, 0.001, input_dim as i32, data_size as i32);

//     println!("");
//     println!("\n Cross : RBF Classification Model : OK");
//     println!("");
//     for i in 0..data_size {
//         let input_ptr: *mut f32 = x[i].as_mut_ptr();
//         let output = predict_rbf_classification(rbf_model, input_ptr);
//         println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
//     }
//     println!("");

//     free_rbf(rbf_model);
// }
