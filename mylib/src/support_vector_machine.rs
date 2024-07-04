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

pub struct SVMModel{
    
}
fn main() {

    // Define problem data
    let p = &[[4.0, 1.0],
        [1.0, 2.0]];
    let q = &[1.0, 1.0];
    let a = &[[1.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]];
    let l = &[1.0, 0.0, 0.0];
    let u = &[1.0, 0.7, 0.7];

    // Extract the upper triangular elements of `P`
    let p = CscMatrix::from(p).into_upper_tri();

    // Disable verbose output
    let settings = Settings::default()
        .verbose(false);

    // Create an OSQP problem
    let mut prob = Problem::new(p, q, a, l, u, &settings).expect("failed to setup problem");

    // Solve problem
    let result = prob.solve();

    // Print the solution
    println!("{:?}", result.x().expect("failed to solve problem"));
}
