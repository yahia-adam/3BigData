use std::ffi::CString;
use mylib::{free_mlp, init_mlp, predict_mlp, train_mlp, MultiLayerPerceptron};

fn main() {
    let x: Vec<Vec<f32>> = vec![
        vec![1.7688197, 1.4136543],
        vec![1.35596694, 1.76459609],
        vec![1.46879524, 1.51093763],
        vec![1.65341619, 1.87900167],
        vec![1.1391278, 1.22748442],
        vec![1.45894563, 1.74506835],
        vec![1.08056787, 1.16916315],
        vec![1.72806726, 1.35750803],
        vec![1.6746088, 1.53563887],
        vec![1.76894957, 1.29643482],
        vec![1.77614841, 1.78068834],
        vec![1.24939396, 1.47060958],
        vec![1.65708247, 1.81335714],
        vec![1.65871942, 1.61564602],
        vec![1.15471803, 1.48532481],
        vec![1.18591832, 1.61386847],
        vec![1.34353248, 1.15704751],
        vec![1.53987207, 1.48039429],
        vec![1.34301121, 1.00791634],
        vec![1.70337313, 1.64219067],
        vec![1.81063802, 1.12171387],
        vec![1.05860799, 1.55813772],
        vec![1.40846516, 1.26497949],
        vec![1.17876654, 1.46114429],
        vec![1.81481273, 1.36591035],
        vec![1.35544183, 1.0415667],
        vec![1.81283123, 1.47640731],
        vec![1.58814081, 1.73597868],
        vec![1.5461268, 1.51031164],
        vec![1.29010061, 1.49248884],
        vec![1.8385353, 1.29573389],
        vec![1.85960634, 1.61225223],
        vec![1.20630309, 1.81995362],
        vec![1.30797264, 1.86572636],
        vec![1.31568402, 1.68974423],
        vec![1.10692199, 1.6971132],
        vec![1.67491044, 1.86053688],
        vec![1.67590324, 1.4606446],
        vec![1.58853217, 1.87616375],
        vec![1.273865, 1.3109698],
        vec![1.41018129, 1.00143566],
        vec![1.13943226, 1.65516981],
        vec![1.48284411, 1.81944692],
        vec![1.33779365, 1.41804992],
        vec![1.56034745, 1.08775547],
        vec![1.05376321, 1.52313113],
        vec![1.21235607, 1.28823511],
        vec![1.13316013, 1.46696951],
        vec![1.35986624, 1.02214684],
        vec![1.88346961, 1.84836095],
        vec![2.72643656, 2.83458674],
        vec![2.86094958, 2.89198648],
        vec![2.60846221, 2.70761465],
        vec![2.74194992, 2.58757516],
        vec![2.15048702, 2.47034506],
        vec![2.23703804, 2.62483117],
        vec![2.28540267, 2.54389726],
        vec![2.73150595, 2.43772649],
        vec![2.66990626, 2.22090982],
        vec![2.51587671, 2.15599423],
        vec![2.58312618, 2.43939136],
        vec![2.15285864, 2.68638132],
        vec![2.22203467, 2.17009891],
        vec![2.39887199, 2.36492678],
        vec![2.7780716, 2.03121604],
        vec![2.44071662, 2.22133016],
        vec![2.54599711, 2.20286037],
        vec![2.50200367, 2.23884246],
        vec![2.74209968, 2.86602696],
        vec![2.33673145, 2.72924101],
        vec![2.26493496, 2.30001389],
        vec![2.89710766, 2.81505498],
        vec![2.20865433, 2.22054359],
        vec![2.27108037, 2.65829914],
        vec![2.36918816, 2.3732643],
        vec![2.50567734, 2.21690975],
        vec![2.67589421, 2.79480049],
        vec![2.69448577, 2.77343251],
        vec![2.58711834, 2.72151332],
        vec![2.61567275, 2.29905194],
        vec![2.40150522, 2.01068073],
        vec![2.6620881, 2.8922813],
        vec![2.15280931, 2.03447378],
        vec![2.05346352, 2.14805805],
        vec![2.64437445, 2.58488613],
        vec![2.01850468, 2.05956394],
        vec![2.54813869, 2.5430906],
        vec![2.62653187, 2.6716472],
        vec![2.06499309, 2.61593904],
        vec![2.42733173, 2.24748975],
        vec![2.51585642, 2.12509329],
        vec![2.89655713, 2.25877066],
        vec![2.8628797, 2.38695392],
        vec![2.76973041, 2.41558355],
        vec![2.02714842, 2.28901157],
        vec![2.33291935, 2.81338416],
        vec![2.70035784, 2.89994242],
        vec![2.37707154, 2.84636671],
        vec![2.65624267, 2.2146111],
        vec![2.56172637, 2.62409468],
    ];
    let y: Vec<f32> = vec![
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        -1.0, -1.0,
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


    let c_log_filename = CString::new("clf_linear_multiple").expect("CString::new failed");
    let c_model_filename = CString::new("../models/examples/mlp/classification/linear_multiple.json").expect("CString::new failed");

    unsafe {
        let success = train_mlp(
            mlp,
            x_train_ptr,
            y_train_ptr,
            data_size as u32,
            x_test_ptr,
            y_test_ptr,
            data_size as u32,
            0.001,
            1_000,
            c_log_filename.as_ptr(),
            c_model_filename.as_ptr(),
            false,
            false,
            false,
        );

        if success {
            let mut correct = 0;
            for i in 0..data_size {
                let input_ptr: *const f32 = x[i].as_ptr();
                let output: *mut f32 = predict_mlp(mlp, input_ptr);
                if !output.is_null() {
                    let res: Vec<f32> = Vec::from_raw_parts(output, 1, 1);
                    // println!("X: {:?}, Y: {:?} ---> MLP model: {:?}", x[i], y[i], res);
                    if (res[0] > 0.0 && y[i] > 0.0) || (res[0] <= 0.0 && y[i] <= 0.0) {
                        correct += 1;
                    }
                }
            }
            let accuracy = correct as f32 / data_size as f32;
            println!("Accuracy: {:.2}%", accuracy * 100.0);

            // let model_file_name = "model.json";
            // let model_file_name = CString::new(model_file_name).expect("CString::new failed");
            // save_mlp_model(mlp, model_file_name.as_ptr());
        } else {
            println!("Training failed.");
        }
        free_mlp(mlp);
    }
}
