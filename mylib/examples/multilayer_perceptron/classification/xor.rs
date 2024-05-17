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

fn main() {

    // datasets
    let sample_inputs: Vec<f32> = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let sample_outputs: Vec<f32> = vec![-1.0, 1.0, 1.0, -1.0];
    let data_size: u32 = 4;
    let inputs: *mut f32 = Vec::leak(sample_inputs).as_mut_ptr();
    let outputs: *mut f32 = Vec::leak(sample_outputs).as_mut_ptr();

    // model init
    let npl: *mut u32 = Vec::leak(vec![2, 3, 1]).as_mut_ptr();
    let model: *mut MultiLayerPerceptron = init_mlp(npl, 3);
    // train model
    train_mlp(model, inputs, outputs, data_size, 0.001, 20000000, true);

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
        println!(
            "Valeur attendu : {:?}, Valeur predit : {:?}",
            outputs[i], result
        );
    }
    free_mlp(model);
}
