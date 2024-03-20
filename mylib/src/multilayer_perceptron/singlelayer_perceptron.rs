/* ********************************************************************************************************* */
/*                                                                                                           */
/*                                                              :::::::::: ::::::::   :::::::: :::::::::::   */
/*   singlelayer_perceptron.rs                                 :+:       :+:    :+: :+:    :+:    :+:        */
/*                                                            +:+       +:+        +:+           +:+         */
/*   By: YA. Adam <adam.y.abdc@gmail.com>                    +#++:++#  +#++:++#++ :#:           +#+          */
/*                                                          +#+              +#+ +#+   +#+#    +#+           */
/*   Created: 2024/03/19 23:27:37 by YA. Adam              #+#       #+#    #+# #+#    #+#    #+#            */
/*   Updated: 2024/03/19 23:27:37 by YA. Adam             ########## ########   ######## ###########         */
/*                                                                                                           */
/* ********************************************************************************************************* */

pub struct SingleLayerPerceptron<T, U> {
    params: Vec<T>,
    results: Vec<U>
}

//  faire des prédictions
pub trait Predict<T, U> {
    fn predict(&self, inputs: &[T]) -> U;
}

impl<T> Predict<T, i32> for SingleLayerPerceptron<T, i32>
where T: std::ops::Mul<Output=T> +std::fmt::Display
{
    fn predict(&self, inputs: &[T]) -> i32 {
        let mut sum = 0;
        for (p,i) in self.params.iter().zip(inputs) {
            println!("{}", i);
        }
        sum
    }
}

fn main()
{
    let sp = SingleLayerPerceptron {
        params: vec![1,2,3],
        results: vec![12]
    };
    let test = vec![1,2,3];
    println!("{}", sp.predict(&test));
}
// // entraîner le modèle sur un ensemble de données d'entraînement
// pub trait Fit<T> {
//     fn fit(&self, datasets: &[T]);
// }
// impl<T> Fit<T> for SingleLayerPerceptron<T> {
//     fn fit(&self, datasets: &[T]) {

//     }
// }

// impl SingleLayerPerceptron {

// }


// pub trait Save {
// // sauvegarder le modèle
//     fn save(&self) -> Sring;
// }

// pub trait Load {
// // charger le model
//     fn load(&self, model_path: str);
// }

// pub trait GetParams {
// // recuperer les paremetres
//     fn get_params<T>(&self) -> [T];
// }

// pub trait SetParams {
// // definir/changer les parametres
//     fn set_params(&self, params: [T]);
// }


// impl Fit for SingleLayerPerceptron {
//     fn fit(&self) {
        
//     }
// }

// impl GetParams for SingleLayerPerceptron {
//     fn get_params(&self) {
//         [1,2]
//     }
// }
    