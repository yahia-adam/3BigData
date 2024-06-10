#[allow(unused_imports)]
use mylib::{SupportVectorClassifier, init_svc, train_svc, predict_svc,free_svc, save_svc, load_svc};

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
    
    let svc: *mut SupportVectorClassifier = init_svc(2);
    train_svc(svc, x_ptr, y_ptr, data_size as u32, 0.01, 10000);
    
    println!("");
    println!("Linear Simple : svc : OK");
    println!("");
    for i in 0..data_size {
        let input_ptr: *mut f32 = Vec::leak(x[i].clone()).as_mut_ptr();
        let output = predict_svc(svc, input_ptr);
        println!("X:{:?}, Y:{:?} ---> mon model: {:?}", x[i], y[i], output);
    }
    println!("");
}
