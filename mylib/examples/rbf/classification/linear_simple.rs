#[allow(unused_imports)]
use mylib::{RadicalBasisFunctionNetwork, init_rbf,
            train_rbf_regression, train_rbf_rosenblatt,
            predict_rbf_regression, predict_rbf_classification,
            free_rbf, save_rbf_model, rbf_to_json };

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
    let data_size = y.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_ptr: *const f32 = Vec::leak(x_flaten.clone()).as_ptr();
    let y_ptr: *const f32 = Vec::leak(y.clone()).as_ptr();
    
    let rbf_model: *mut RadicalBasisFunctionNetwork = init_rbf(data_size, 2, 2);
    train_rbf_regression(rbf_model, x_ptr, y_ptr, data_size as u32, 0.01, 1000);
    
    println!("");
    println!("Linear Simple : Linear Model : OK");
    println!("");
    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_linear_model(linear_model, input_ptr);
        println!("X:{:?}, Y:{:?} ---> mon model: {:?}", x[i], y[i], output);
    }
    println!("");
}
