#[allow(unused_imports)]
use mylib::{
    RadicalBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt,
    predict_rbf_classification, free_rbf, save_rbf_model, rbf_to_json,
};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
        vec![0.25, 0.75],
    ];
    let y: Vec<f32> = vec![
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0
    ];
    let data_size = y.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_ptr: *const f32 = Vec::leak(x_flaten.clone()).as_ptr();
    let y_ptr: *const f32 = Vec::leak(y.clone()).as_ptr();

    let input_dim = 2;
    let cluster_num = 6;
    let gamma = 1.0;

    let rbf_model: *mut RadicalBasisFunctionNetwork = init_rbf(input_dim, cluster_num, gamma);

    let iterations_count = 1000;
    let alpha = 0.01;

    train_rbf_rosenblatt(rbf_model, x_ptr as *mut f32, y_ptr as *mut f32, iterations_count, alpha, input_dim, data_size as i32);

    println!("");
    println!("\n RBF Multilinear 3 Classes Classification : OK");
    println!("");
    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_rbf_classification(rbf_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
    }
    println!("");

    free_rbf(rbf_model);
}