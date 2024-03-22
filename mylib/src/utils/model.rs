//  faire des prédictions
mod Model {
    pub trait Predict
    {
        extern "C" fn guess(&self, inputs: &[f32]) -> f32;
        extern "C" fn predict(&self, inputs: &[f32]);
    }
    
    // Entraîner le modèle sur un ensemble de données d'entraînement
    pub trait Fit
    {
        extern "C" fn fit(&mut self, datasets: (&Vec<Vec<f32>>, &Vec<f32>), learning_rate: f32, epochs: u32, tolerance: f32);
    }
    
    pub trait GetWeights
    {
        extern "C" fn get_weights(&self) -> Vec<f32>;
    }
    
    pub trait SetWeights
    {
        extern "C" fn set_weights(&mut self, weights: Vec<f32>);
    }
    
    // afficher les poids du models
    pub trait ToString
    {
        extern "C" fn to_string(&self) -> String;
    }
    
    pub trait GetHistory
    {
        extern "C" fn get_history(&self) -> Vec<f32>;
    }
    
    pub trait Save {
    // sauvegarder le modèle
        
        extern "C" fn save(&self, filepath: &str) ->std::io::Result<()>;
    }
    
    pub trait Load {
    // charger le model
        extern "C" fn load(&mut self, filepath: &str);
    }
}
