#[allow(unused_imports)]
use mylib::{
    init_linear_model, load_linear_model, predict_linear_model, save_linear_model,
    train_linear_model, LinearModel, generate_dataset
};


fn main() {
    let (x, y) = generate_dataset();
    let data_size = y.len();

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect::<Vec<f32>>();
    let x_ptr: *const f32 = Vec::leak(x_flaten.clone()).as_ptr();
    let y_ptr: *const f32 = Vec::leak(y.clone()).as_ptr();
    
    let linear_model: *mut LinearModel = init_linear_model(2, true);
    train_linear_model(linear_model, x_ptr, y_ptr, data_size as u32, 0.01, 100000);

    println!("");
    println!("\n Cross : Linear Model    : KO");
    println!("");
    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_linear_model(linear_model, input_ptr);
        println!("X:{:?}, Y:[{:?}] ---> Linear Model model: {:?}", x[i], y[i], output);
    }
    println!("");
}
