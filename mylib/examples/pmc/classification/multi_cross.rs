#[allow(unused_imports)]
use mylib::{MultiLayerPerceptron , init_mlp, train_mlp, predict_mlp, free_mlp, save_mlp_model};


fn main() {

    let data_size = y.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_ptr: *mut f32 = Vec::leak(x_flaten.clone()).as_mut_ptr();
    let y_ptr: *mut f32 = Vec::leak(y.clone()).as_mut_ptr();
    
    let mut npl = vec![2,1];

    let mlp: *mut MultiLayerPerceptron = init_mlp(npl.as_mut_ptr(), 2, true);
    train_mlp(mlp, x_ptr, y_ptr, data_size as u32, 0.001, 1000000);
    
    println!("");
    println!("\n Linear multiple : pmc : OK");
    println!("");
    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_mlp(mlp, input_ptr);
        
        let res: Vec<f32> =
        unsafe { Vec::from_raw_parts(output, 1, 1) };

        println!("X:{:?}, Y:{:?} ---> MLP model: {:?}", x[i], y[i], res);
    }
    println!("");
    
}
