#[allow(unused_imports)]
use mylib::{RadicalBasisFunctionNetwork, init_rbf, train_rbf_regression, predict_rbf_regression, free_rbf};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
        vec![4.0],
        vec![5.0],
    ];
    let y: Vec<f32> = vec![
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ];
    let sample_count = y.len() as i32;
    let input_dim = 1;
    let cluster_num = 3;
    let gamma = 0.1;

    let mut x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let x_ptr: *mut f32 = x_flatten.as_mut_ptr();
    let y_ptr: *mut f32 = y.clone().as_mut_ptr();

    let rbf_model: *mut RadicalBasisFunctionNetwork = init_rbf(input_dim, cluster_num, gamma);
    train_rbf_regression(rbf_model, x_ptr, y_ptr, input_dim, sample_count);

    println!("");
    println!("\n RBF Regression 2D : RBF Model : OK");
    for i in 0..sample_count as usize {
        let input_ptr: *mut f32 = x[i].clone().as_mut_ptr();
        let output = predict_rbf_regression(rbf_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
    }
    println!("");

    free_rbf(rbf_model);
}