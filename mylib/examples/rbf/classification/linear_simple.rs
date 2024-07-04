#[allow(unused_imports)]
use mylib::{RadicalBasisFunctionNetwork, init_rbf, train_rbf_rosenblatt,
            predict_rbf_classification, free_rbf, save_rbf_model, rbf_to_json};

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

    let rbf_class_model: *mut RadicalBasisFunctionNetwork = init_rbf(input_dim, cluster_num, gamma);
    train_rbf_rosenblatt(rbf_class_model, x_ptr, y_ptr, 1000, 0.01, input_dim, sample_count);

    println!("RBF Classification Model: OK");
    println!("");
    for i in 0..sample_count as usize {
        let input_ptr: *mut f32 = x[i].clone().as_mut_ptr();
        let output = predict_rbf_classification(rbf_class_model, input_ptr);
        println!("X: {:?}, Y: {:?} ---> RBF classification: {:?}", x[i], y[i], output);
    }
    println!("");

    let loaded_model = rbf_to_json(b"rbf_model.json\0".as_ptr() as *const i8);

    free_rbf(rbf_class_model);
    free_rbf(loaded_model);
}