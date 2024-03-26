#[cfg(test)]
mod tests {
    use std::ffi::{c_char, CString};
    use mylib::linear_model::{get_weights, guess, new, set_weights, save_model, load_model, fit, LinearRegression};

    #[test]
    fn liner_regression_get_weights() {
        let model1: *mut LinearRegression = new();
        let model2: *mut LinearRegression =  new();
        assert_eq!(get_weights(model1), get_weights(model2));
    }

    #[test]
    fn create_liner_regression() {
        let model: *mut LinearRegression = new();
        
        let model2: LinearRegression = LinearRegression {
            weights: vec![],
            activate: "identity".to_string()
        };
    
        let boxed_model: Box<LinearRegression> = Box::new(model2);
        let leaked_boxed_model: *mut LinearRegression = Box::leak(boxed_model);

        assert_eq!(get_weights(leaked_boxed_model), get_weights(model));
    }

    #[test]
    fn linear_model_set_weights() {
        let model: *mut LinearRegression = new();
        let weights = vec![1.0, 2.0, 3.0];
        set_weights(model, weights);
        assert_eq!(get_weights(model), vec![1.0, 2.0, 3.0])
    }

    #[test]
    fn linear_model_guess(){
        let model: *mut LinearRegression = new();
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        set_weights(model, weights);
        let inputs: Vec<f32> = vec![2.0, 3.0];
        assert_eq!(guess(model, inputs) as i32, 14);
    }

    #[test]
    fn linear_model_to_string_save_loads() {
        let model: *mut LinearRegression = new();
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        set_weights(model, weights);
        
        let filepath_cstr: CString = CString::new("./tests/test_linear_model.json").expect("Failed to create CString");
        let filepath_ptr: *const c_char = filepath_cstr.as_ptr();
        save_model(model, filepath_ptr);

        let model2 = load_model(filepath_ptr);
        let inputs: Vec<f32> = vec![2.0, 3.0];
        let inputs2: Vec<f32> = vec![2.0, 3.0];

        assert_eq!(guess(model, inputs) as i32, guess(model2, inputs2) as i32);
    }

    fn celsius_to_fahrenheit(value: f32) -> f32 {
        value * 1.8 + 32.0
    }

    #[test]
    fn linear_model_fit() {

        let sp:*mut LinearRegression = new();

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

        fit(sp,
            leaked_celsius_temperatures,
            celsius_temperatures_row,
            celsius_temperatures_col,
            leaked_fahrenheit_temperatures,
            fahrenheit_temperatures_row,
            0.001,
            100);
            
        assert_eq!(guess(sp, vec![96.0]) as i32, 204)
    }

}