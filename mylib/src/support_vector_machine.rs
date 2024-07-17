/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   support_vector_machine.rs                                 :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YAHIA ABDCHAFAA Adam, SALHAB Charbel, ELOY Theo     +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/22 19:38:54                          #+#       #+#    #+# #+#    #+#    #+#            */
/*   3IABD1 2023-2024                                     ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

use osqp::{CscMatrix, Problem, Settings};

pub struct SVMModel {
    weight: Vec<f32>,
    biais: f32,
    sample_size: u32,
    sample_len: u32,
    kernel: String,
    deg: i32,
}

fn main() {

    // Define problem data
    /*let P = &[[4.0, 1.0],
        [1.0, 2.0]];
    let q = &[1.0, 1.0];
    let A = &[[1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]];
    let l = &[1.0, 0.0, 0.0];
    let u = &[1.0, 0.7, 0.7];

    // Extract the upper triangular elements of `P`
    let P = CscMatrix::from(P);

    // Disable verbose output
    let settings = Settings::default()
        .verbose(false);

    // Create an OSQP problem
    let mut prob = Problem::new(P, q, A, l, u, &settings).expect("failed to setup problem");

    // Solve problem
    let result = prob.solve();

    // Print the solution
    println!("{:?}", result.x().expect("failed to solve problem"));*/


    let mut model = SVMModel {
        weight: Vec::new(),
        biais: 0.0,
        sample_size: 3,
        sample_len: 9,
        deg: 2,
        kernel: "linear".parse().unwrap(),
    };


    let x1: Vec<Vec<f32>> = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0]
    ];
    let y1: Vec<f32> = vec![
        1.0,
        -1.0,
        -1.0
    ];


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

    train(model, &x1, &y1, &2.0, y1.len() as u32, x1[0].len() as u32);

}

fn get_kernel(model: &SVMModel, xi: &Vec<f32>, xj: &Vec<f32>) -> f32 {
    if model.kernel == "linear" {
        //poduit scalaire
        xi.iter().zip(xj.iter()).map(|(i, j)| i * j).sum()
    } /*else if model.kernel == "poly" {
        f32::powi(1.0 + xi * xj, model.deg)
    } else if model.kernel == "rad" {
        expf(-f32::powi(xi, 2)) * expf(-f32::powi(xj, 2)) * expf(2.0 * xi * xj)
    }
    */else { 0.0 }
}


fn train(mut model: SVMModel, inputs: &Vec<Vec<f32>>, labels: &Vec<f32>, gamma: &f32, input_length: u32, dimensions: u32) {
    let dimensions = inputs[1].len() as usize;
    let input_length = labels.len() as usize;


    let mut big_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; dimensions + input_length + 1];
    for i in 0..dimensions {
        big_matrix[i][i] = 1f64;
    }

    //println!("P: {:?}", big_matrix);

    let big_csc_matrix = CscMatrix::from(&big_matrix).into_upper_tri();
    println!("P csc: {:?}", big_csc_matrix);

    let mut q: Vec<f64> = vec![0f64; dimensions + 1 + input_length];
    for i in 0..input_length {
        q[dimensions + 1 + i] = 1f64;
    }
    //println!("q :{:?}", q);

    let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; 2 * input_length];
    for row in 0..input_length {
        for col in 0..dimensions {
            a_matrix[row][col] = (labels[row] * inputs[row][col]) as f64
        }
        a_matrix[row][dimensions] = labels[row] as f64;
        a_matrix[row][row + dimensions + 1] = -1f64;
        a_matrix[row + input_length][row + dimensions + 1] = 1f64;
    }
    let mut a_matrix: Vec<Vec<f64>> = vec![vec![0f64; dimensions + input_length + 1]; 2 * input_length];
    for row in 0..input_length {
        for col in 0..dimensions {
            a_matrix[row][col] = (labels[row] * inputs[row][col]) as f64;
        }
        a_matrix[row][dimensions] = labels[row] as f64;
        a_matrix[row][row + dimensions + 1] = -1f64;
        a_matrix[row + input_length][row + dimensions + 1] = 1f64;
    }


    //println!("a :{:?}", a_matrix);
    let a_csc_matrix = CscMatrix::from(&a_matrix);
    println!("a_csc :{:?}", a_csc_matrix);

    let mut l: Vec<f64> = Vec::new();
    l.extend(vec![1f64; input_length]);
    l.extend(vec![0f64; input_length]);
    println!("l: {:?}", l);

    let mut u: Vec<f64> = vec![f64::INFINITY; 2 * input_length];
    println!("u: {:?}", u);

    let settings = Settings::default().verbose(true);
    let mut problem = Problem::new(&big_csc_matrix, &*q, &a_matrix, &*l, &*u, &settings).expect("OSQP Setup Error");

    // Solve problem
    let result = problem.solve();
    let alphas = result.x().expect("failed to solve problem");

    // Print the solution
    println!("alphas {:?}", alphas);


    let w: Vec<f32> = (0..dimensions)
        .map(
            |dim|
            alphas.iter()
                .zip(labels)
                .zip(inputs)
                .map(
                    |((alpha, label), input)|
                    *alpha as f32 * *label * input[dim])
                .sum::<f32>())
        .collect();

    println!("weights: {:?}", w);


    let support_vector = alphas.iter().position(|alpha| *alpha >= 1e-6).unwrap();
    println!("sup_vec: {}", support_vector);

    let bias =
        1f32 / labels[support_vector]
            + inputs[support_vector].iter()
            .map(|i| i * alphas[support_vector] as f32)
            .sum::<f32>();

    println!("bias: {}", bias);
    model.biais = bias;
    model.weight = w;

    println!();
    for i in 0..input_length {
        predict_svm(&model, &inputs[i], labels[i]);
    }
}

pub fn predict_svm(model: &SVMModel, inputs: &Vec<f32>, label: f32) -> f32 {
    let result: f32 = inputs.iter().zip(&model.weight).map(|(i, w)| w * i + model.biais).sum::<f32>();

    println!("result {:?}({}): {}", inputs, label, result);
    result
}
