#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt,
            predict_rbf_classification, free_rbf};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0]
    ];
    let y: Vec<f32> = vec![
        1.0,
        -1.0,
        -1.0
    ];
    let sample_count = y.len() as i32;
    let input_dim = x[0].len() as i32;
    let cluster_num = 2;
    let gamma = 1.0;

    let mut x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let x_ptr: *mut f32 = x_flatten.as_mut_ptr();
    let y_ptr: *mut f32 = y.clone().as_mut_ptr();

    let rbf_class_model: *mut RadialBasisFunctionNetwork = init_rbf(input_dim, cluster_num, gamma);
    train_rbf_rosenblatt(rbf_class_model, x_ptr, y_ptr, 100, 0.01, input_dim, sample_count);

    println!("\n RBF Classification Model: OK");
    println!("");
    for i in 0..sample_count as usize {
        let input_ptr: *mut f32 = x[i].clone().as_mut_ptr();
        let output = predict_rbf_classification(rbf_class_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF classification: {:?}", x[i], y[i], output);
    }
    println!("");

    free_rbf(rbf_class_model);
}