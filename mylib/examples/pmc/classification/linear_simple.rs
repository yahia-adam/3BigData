#[allow(unused_imports)]
use mylib::{MultiLayerPerceptron, init_mlp, train_mlp, predict_mlp, free_mlp, save_mlp_model};

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

    let x_flaten: Vec<f32> = x.clone().into_iter().flatten().collect();
    let y_train: Vec<f32> = y.clone();

    let x_train_ptr: *const f32 = x_flaten.as_ptr();
    let y_train_ptr: *const f32 = y_train.as_ptr();

    let x_test_ptr: *const f32 = x_flaten.as_ptr();
    let y_test_ptr: *const f32 = y_train.as_ptr();

    let npl: Vec<u32> = vec![2, 1];
    let mlp: *mut MultiLayerPerceptron = init_mlp(npl.as_ptr(), npl.len() as u32, true);
    
    unsafe {
        let success = train_mlp(
            mlp,
            x_train_ptr,
            y_train_ptr,
            data_size as u32,
            x_test_ptr,
            y_test_ptr,
            data_size as u32,
            0.0001,
            10000
        );
        
        if success {
            println!("\nLinear Simple : PMC : OK\n");

            for i in 0..data_size {
                let input_ptr: *const f32 = x[i].as_ptr();
                let mut output_size: usize = 0;
                let output: *mut f32 = predict_mlp(mlp, input_ptr, &mut output_size);

                if !output.is_null() {
                    let res: Vec<f32> = Vec::from_raw_parts(output, output_size, output_size);
                    println!("X: {:?}, Y: {:?} ---> MLP model: {:?}", x[i], y[i], res);
                }
            }

            save_mlp_model(mlp, std::ffi::CString::new("model.json").unwrap().as_ptr());
        } else {
            println!("Training failed.");
        }
        free_mlp(mlp);
    }
}
