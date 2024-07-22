use std::{fs, path::PathBuf};

use mylib::create_serialized_ml_dataset;

const BASE_DIR: &str = "../dataset";
const OUTPUT_DIR: &str = "../serialized_datasets";

fn create_all_serialized_datasets() -> std::io::Result<()> {
    let base_dir = PathBuf::from(BASE_DIR);
    
    // Créer le répertoire de sortie s'il n'existe pas
    fs::create_dir_all(OUTPUT_DIR)?;

    // Metal vs Other
    create_serialized_ml_dataset(
        base_dir.to_str().unwrap(),
        &format!("{}/metal_vs_other.bin", OUTPUT_DIR),
        1.0, -1.0, -1.0
    )?;
    println!("metal_vs_other");

    // Paper vs Other
    create_serialized_ml_dataset(
        base_dir.to_str().unwrap(),
        &format!("{}/paper_vs_other.bin", OUTPUT_DIR),
        -1.0, 1.0, -1.0
    )?;
    println!("paper_vs_other");

    // Plastic vs Other
    create_serialized_ml_dataset(
        base_dir.to_str().unwrap(),
        &format!("{}/plastic_vs_other.bin", OUTPUT_DIR),
        -1.0, -1.0, 1.0
    )?;
    println!("plastic_vs_other");

    println!("All serialized datasets created successfully.");
    Ok(())
}

fn main() {
    if let Err(e) = create_all_serialized_datasets() {
        eprintln!("Error creating serialized datasets: {}", e);
    }
}