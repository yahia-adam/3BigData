#[allow(unused_imports)]
use mylib::{RadialBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt, predict_rbf_classification, free_rbf, generate_dataset};

fn main() {

    let (x, y) = generate_dataset();
    let sample_count = y.len() as i32;
    let input_dim = x[0].len() as i32;
    let cluster_num = 10;
    let gamma = 1.0;

    let mut x_flatten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let x_ptr: *mut f32 = x_flatten.as_mut_ptr();
    let y_ptr: *mut f32 = y.as_mut_ptr();

    let rbf_model: *mut RadialBasisFunctionNetwork = init_rbf(input_dim, cluster_num, gamma);

    train_rbf_rosenblatt(rbf_model, x_ptr, y_ptr, 1000, 0.01, input_dim, sample_count);

    println!("");
    println!("\n RBF Classification Model : OK");
    println!("");
    for i in 0..sample_count as usize {
        let input_ptr: *mut f32 = x[i].as_mut_ptr();
        let output = predict_rbf_classification(rbf_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF model: {:?}", x[i], y[i], output);
    }
    println!("");

    free_rbf(rbf_model);
}