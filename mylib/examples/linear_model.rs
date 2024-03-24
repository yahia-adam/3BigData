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
// extern crate mylib;
use mylib::linear_model::LinearModel;

fn celsius_to_fahrenheit(value: f32) -> f32 {
    value * 1.8 + 32.0
}

fn main()
{
    // let name = env::args().skip(1).next();
    // println!("Hello, {}!", name.unwrap_or("world".into()));
    let mut sp = LinearModel::new();

    let celsius_temperatures: Vec<f32> = vec![ 0.0, 11.0, 27.0, 6.0, 13.0, 39.0, 19.0, 25.0, 30.0, -15.0, 14.0, 33.0, -5.0, 20.0, 22.0, -10.0, 17.0, 37.0, 24.0, -20.0, 8.0, 35.0, -30.0, 21.0, 10.0, -25.0, -2.0, -40.0, 16.0, 29.0, 7.0, -35.0, 36.0, -22.0, -38.0, 28.0, -45.0, 38.0, 3.0, 18.0, -12.0, -18.0, 32.0, -8.0, 15.0, 2.0, -33.0, 26.0, -50.0, 12.0, -3.0, 5.0];
    let celsius_temperatures_row: usize = celsius_temperatures.len();
    let celsius_temperatures_col:usize = 1;

    let mut fahrenheit_temperatures: Vec<f32> = vec![];
    for i in &celsius_temperatures {
        fahrenheit_temperatures.push(celsius_to_fahrenheit(*i));
    }

    let fahrenheit_temperatures_row: usize = fahrenheit_temperatures.len();
    let leaked_celsius_temperatures = Vec::leak(celsius_temperatures).as_ptr();

    let leaked_fahrenheit_temperatures = Vec::leak(fahrenheit_temperatures).as_ptr();

    sp.fit(leaked_celsius_temperatures,
        celsius_temperatures_row,
        celsius_temperatures_col,
        leaked_fahrenheit_temperatures,
        fahrenheit_temperatures_row,
        0.001,
        100);

        for i in 0..=100 {
            let celsius = i as f32;
            println!("{}C = prdicted_value={}, expected_value={}", celsius, sp.guess(vec![celsius]), celsius_to_fahrenheit(celsius));
        }


}
