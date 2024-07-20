#[allow(unused_imports)]
use mylib::{
    RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt,
    predict_rbf_classification, free_rbf, save_rbf_model
};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y: Vec<f32> = vec![
        1.0,
        1.0,
        -1.0,
        -1.0
    ];
    let data_size = y.len();
    let input_dim = x[0].len();

    let x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_ptr: *mut f32 = Vec::leak(x_flatten.clone()).as_mut_ptr();
    let y_ptr: *mut f32 = Vec::leak(y.clone()).as_mut_ptr();

    let cluster_num = 4;
    let gamma = 2.0;

    let rbf_model: *mut RadialBasisFunctionNetwork = init_rbf(input_dim as i32, cluster_num, gamma);
    train_rbf_rosenblatt(rbf_model, x_ptr, y_ptr, 1000, 0.01, input_dim as i32, data_size as i32);

    println!("");
    println!("\n XOR : RBF Model    : OK");
    println!("");

    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_rbf_classification(rbf_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
    }
    println!("");


    free_rbf(rbf_model);
}