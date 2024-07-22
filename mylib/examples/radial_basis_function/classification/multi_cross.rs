// #[allow(unused_imports)]
// use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf};
// use mylib::generate_dataset;

// fn main() {
//     let (mut x, mut y) = generate_dataset();
//     let data_size = y.len();
//     let input_dim = x[0].len();

//     let x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
//     let x_ptr: *mut f32 = Vec::leak(x_flatten.clone()).as_mut_ptr();
//     let y_ptr: *mut f32 = Vec::leak(y.clone()).as_mut_ptr();

//     let cluster_num = 32;
//     let gamma = 1.0;

//     let rbf: *mut RadialBasisFunctionNetwork = init_rbf(input_dim as i32, cluster_num, gamma);
//     train_rbf_rosenblatt(rbf, x_ptr, y_ptr, 10000000, 0.001, input_dim as i32, data_size as i32);

//     println!("");
//     println!("\n Multi Cross : RBF Classification : KO");
//     println!("");
//     for i in 0..data_size {
//         let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
//         let output = predict_rbf_classification(rbf, input_ptr);
//         println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
//     }
//     println!("");

//     free_rbf(rbf);
// }