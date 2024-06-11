<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use FFI;

class linearModel extends Command
{
    /**
     * The name and signature of the console command.
     *
     * @var string
     */
    protected $signature = 'app:linear-model';

    /**
     * The console command description.
     *
     * @var string
     */
    protected $description = 'Train a linear model form datasets';

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $libpath = env('LIB_PATH', '../mylib/target/release/libmylib.so');

        $ffi = FFI::cdef("
            typedef struct LinearModel LinearModel;
            LinearModel* init_linear_model(unsigned int input_count, bool is_classification);
            void train_linear_model(LinearModel* model, const float* features, const float* labels, unsigned int data_size, float learning_rate, unsigned int epochs);
            float predict_linear_model(LinearModel* model, float* inputs);
            const char* to_json(const LinearModel* model);
            void save_linear_model(const LinearModel* model, const char* filepath);
            LinearModel* load_linear_model(const char* json_str_ptr);
            void free_linear_model(LinearModel* model);
        ", $libpath);

        try {
            
            $X = FFI::new("float[6]", false, true);
            $Y = FFI::new("float[3]", false, true);
            
            $X[0] = 1.0;
            $X[1] = 1.0;
            $X[2] = 2.0;
            $X[3] = 3.0;
            $X[4] = 3.0;
            $X[5] = 3.0;
            
            $Y[0] = 1.0;
            $Y[1] = -1.0;
            $Y[2] = -1.0;
            
            $data_size = 3;

            $predictions = [];
            $testInputs = [
                [1.0, 1.0],
                [2.0, 3.0],
                [3.0, 3.0]
            ];

            $model = $ffi->init_linear_model(2, true);
            $ffi->train_linear_model($model, $X, $Y, $data_size, 0.001, 100000);
            $testInput = FFI::new("float[2]", false, true);

            foreach ($testInputs as $input) {
                $testInput[0] = $input[0];
                $testInput[1] = $input[1];
                $predictions[] = $ffi->predict_linear_model($model, $testInput);
            }

            foreach ($predictions as $prediction) {
                $this->info("Prediction: " . $prediction);
            }
            $ffi->free_linear_model($model);
            
        } catch (\Throwable $e) {
            $this->error("An error occurred: " . $e->getMessage());
            return 1;
        }

        $this->info('Model training completed successfully.');
        return 0;
    }
}
