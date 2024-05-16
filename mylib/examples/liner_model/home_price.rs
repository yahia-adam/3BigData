
use std::{collections::vec_deque, ffi::c_float};

/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   linearModel.rs                                            :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 14:20:18 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/22 14:20:18 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */
#[allow(unused_imports)]
use mylib::{init_linear_model, train_linear_model, predict_linear_model, save_linear_model, load_linear_model, LinearModel};

fn celsius_to_fahrenheit(value: f32) -> f32 {
    value * 1.8 + 32.0
}

fn main()
{

        let sp:*mut LinearModel = init_linear_model(2);

        // Create and init model
        let features: Vec<f32> = vec![
            100.0, 3.0,
            150.0, 4.0,
            200.0, 5.0,
            120.0, 3.0,
            80.0, 2.0,
            90.0, 2.0,
            110.0, 3.0,
            140.0, 4.0,
            130.0, 3.0,
            160.0, 4.0,
            70.0, 1.0,
            180.0, 5.0,
            170.0, 4.0,
            95.0, 2.0,
            85.0, 2.0
        ];
        let labels: Vec<f32> = vec![
            250.0, 350.0, 450.0, 270.0, 180.0, 190.0, 240.0, 330.0, 290.0, 370.0, 160.0, 420.0, 380.0, 200.0, 175.0
        ];
        let data_size: usize = 15;
        let features_ptr: *const f32 = Vec::leak(features).as_ptr();
        let labels_ptr: *const f32 = Vec::leak(labels.clone()).as_ptr();

        train_linear_model(sp,
            features_ptr,
            labels_ptr,
            data_size as u32,
            0.00001,
            1000);

        let tests_features: Vec<Vec<c_float>> = vec![
            vec![100.0, 3.0],//250.0,
            vec![150.0, 4.0],//350.0,
            vec![200.0, 5.0],//450.0,
            vec![120.0, 3.0],//270.0,
            vec![80.0, 2.0],//180.0,
            vec![90.0, 2.0],//190.0,
            vec![110.0, 3.0],//240.0
            vec![140.0, 4.0],//330.0,
            vec![130.0, 3.0],//290.0
            vec![160.0, 4.0],//370.0
            vec![70.0, 1.0],//160.0
            vec![180.0, 5.0],//420.0,
            vec![170.0, 4.0],//380.0
            vec![95.0, 2.0],//200.0
            vec![85.0, 2.0]//175.0
        ];

        for i in 0..tests_features.len() {
            let input: *mut f32 = Vec::leak(tests_features[i].clone()).as_mut_ptr();
            println!("prdicted_value={}, expected_value={}", predict_linear_model(sp, input), labels[i] );
        }

}
