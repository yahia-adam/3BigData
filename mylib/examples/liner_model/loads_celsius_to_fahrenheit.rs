/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   loads_celsius_to_fahrenheit.rs                            :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/24 22:25:09 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/24 22:25:09 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

use mylib::linear_model::{LinearRegression, load_model, predict};
use std::ffi::{c_char, CString};

fn celsius_to_fahrenheit(value: f32) -> f32 {
    value * 1.8 + 32.0
}

fn main()
{
    let filepath_cstr: CString = CString::new("../models/linear_model/celsius_to_fahrenheit.json").expect("Failed to create CString");
    let filepath_ptr: *const c_char = filepath_cstr.as_ptr();
    let sp: *mut LinearRegression = load_model(filepath_ptr);
    for i in 0..=100 {
        let leak_celsius: *const f32 = Vec::leak(vec![i as f32]).as_ptr();
        println!("prdicted_value = {}, expected_value = {}", predict(sp, leak_celsius, 1), celsius_to_fahrenheit(i as f32));
    }
}