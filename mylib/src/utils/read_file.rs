// use std::error::Error;
// use csv::Reader;

// pub fn read_csv(filepath: &str)  -> Result<(), Box<dyn Error>> {
//     let mut rdr = Reader::from_path(filepath)?;
//     for result in rdr.records() {
//         let record = result?;
//         println!("{:?}", record);
//     }
//     Ok(())
// }

// fn main() {
//     read_csv("../../../datasets/celsius_fahrenheit/datasets.csv");
// }