/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   multilayer_perceptron.rs                                  :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 14:20:22 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 14:20:22 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

#[allow(unused_imports)]
use mylib::{free_mlp, init_mlp, predict_mlp, save_mlp_model, train_mlp, MultiLayerPerceptron};
use std::{
    env,
    ffi::{c_char, CString},
    vec,
};

fn main() {
    let model_path: Option<String> = env::args().nth(1);
    if let Some(path) = model_path {
        // file to save models
        let filepath_cstr: CString = CString::new(path).expect("Failed to create CString");
        let filepath_ptr: *const c_char = filepath_cstr.as_ptr();

        // datasets
        let sample_inputs: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let sample_outputs: Vec<f32> = vec![-1.0, 1.0, 1.0, -1.0];
        let row: usize = 4;
        let input_col: usize = 2;
        let output_col: usize = 1;
        let inputs: *mut f32 = Vec::leak(sample_inputs).as_mut_ptr();
        let outputs: *mut f32 = Vec::leak(sample_outputs).as_mut_ptr();

        // model init
        let npl: *mut usize = Vec::leak(vec![2, 3, 1]).as_mut_ptr();
        let model: *mut MultiLayerPerceptron = init_mlp(npl, 3);

        save_mlp_model(model, filepath_ptr);
        // train model
        train_mlp(
            model, inputs, outputs, input_col, output_col, row, 0.1, 200000, true,
        );
        
        let tests: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let outputs: Vec<f32> = vec![-1.0, 1.0, 1.0, -1.0];

        println!("\n<----------------- test ---------------------->");        
        for i in 0..tests.len() {
            let test_ptr: *mut f32 = Vec::leak(tests[i].clone()).as_mut_ptr();
            let res_ptr: *mut f32 = predict_mlp(model, test_ptr, tests[i].len(), true);
            let result: Vec<f32> = unsafe { Vec::from_raw_parts(res_ptr, 1, 1) };
            println!("Valeur attendu : {:?}, Valeur predit : {:?}", outputs[i], result);
        }
        free_mlp(model);
    }
}
