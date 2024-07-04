#[allow(unused_imports)]
use mylib::{MultiLayerPerceptron , init_mlp, train_mlp, predict_mlp, free_mlp, save_mlp_model};


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
    let x_ptr: *mut f32 = Vec::leak(x_flaten.clone()).as_mut_ptr();
    let y_ptr: *mut f32 = Vec::leak(y.clone()).as_mut_ptr();
    
    let mut npl: Vec<u32> = vec![2,3,2];

    let mlp: *mut MultiLayerPerceptron = init_mlp(npl.as_mut_ptr(), 3, true);
    // train_mlp(mlp, x_ptr, y_ptr, data_size as u32, 0.001, 1_000_000);
    
    // println!("");
    // println!("Xor : pmc : OK");
    // println!("");
    // for i in 0..data_size {
    //     let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
    //     let output = predict_mlp(mlp, input_ptr);
        
    //     let res: Vec<f32> =
    //     unsafe { Vec::from_raw_parts(output, 1, 1) };

    //     println!("X:{:?}, Y:{:?} ---> MLP model: {:?}", x[i], y[i], res);
    // }
    // println!("");
}
