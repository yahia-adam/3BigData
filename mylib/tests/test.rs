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
        };
    
        let boxed_model: Box<LinearRegression> = Box::new(model2);
        let leaked_boxed_model: *mut LinearRegression = Box::leak(boxed_model);

        assert_eq!(get_weights(leaked_boxed_model), get_weights(model));
    }

    #[test]
    fn linear_model_set_weights() {
        let model: *mut LinearRegression = new();
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
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


// <------------------------------------------------------------------------------------------ // multiperceptron -------------------------------------------------------------------------------------->

    // #[test]
    // fn test_propagation() {
    //     let npl: Vec<i32> = vec![2, 3, 1]; // Exemple de réseau avec 2 entrées, une couche cachée de 3 neurones, et 1 sortie
    //     let mut mlp: mylib::multilayer_perceptron::MultiLayerPerceptron = init_mlp(npl);
    //     let inputs: Vec<f64> = vec![0.5, -0.5]; // Données d'entrée simulées
    //     mlp = propagate(mlp, inputs, true);

    //     // Vérifier si les activations sont non nulles (ou toute autre assertion spécifique)
    //     assert!(mlp.x[1].iter().all(|&x| x != 0.0));
    // }
    // #[test]
    // fn test_backpropagation_and_weight_update() {
    //     let npl: Vec<i32> = vec![2, 2, 1];
    //     let mut mlp = init_mlp(npl);
    //     let inputs: Vec<f64> = vec![0.5, -0.5];
    //     let outputs: Vec<f64> = vec![1.0]; // Sortie attendue

    //     mlp = propagate(mlp, inputs, false);
    //     backpropagate(&mut mlp, &outputs, false);
    //     update_weights(&mut mlp, 0.1);

    //     // Vérifier que les poids ont été modifiés
    //     assert_ne!(mlp.weights[1][0][1], 0.0); // Vérifier que le poids a changé
    // }
    // #[test]
    // fn test_integration() {
    //     let npl = vec![3, 4, 2]; // Configuration simple
    //     let mut mlp = init_mlp(npl);
    //     let all_inputs = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    //     let all_outputs = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    //     let alpha = 0.05;
    
    //     for _ in 0..100 { // Un petit nombre d'itérations pour le test
    //         mlp_train(mlp.clone(), all_inputs.clone(), all_outputs.clone(), alpha, 100, true);
    //     }
    
    //     // Vérifier que le modèle donne des prédictions raisonnables
    //     let test_inputs = vec![0.1, 0.2, 0.3];
    //     let prediction = predict(mlp, test_inputs, true);
    //     assert!(prediction[0] > prediction[1]); // Vérifier une condition basée sur votre logique d'affaires
    // }
    
}