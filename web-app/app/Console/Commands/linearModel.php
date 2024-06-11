<?php

namespace App\Console\Commands;

use Illuminate\Console\Command;
use App\Providers\DatasetServiceProvider;
use App\Services\DatasetService;
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
    protected $datasetService;

    public function __construct(DatasetService $datasetService)
    {
        parent::__construct();  // Call the parent constructor
        $this->datasetService = $datasetService;
    }

    /**
     * Execute the console command.
     */
    public function handle()
    {
        $libpath = env('LIB_PATH', '../mylib/target/release/libmylib.so');
        $this->datasetService->init_dataset();

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
            $model = $ffi->init_linear_model($this->datasetService->x_size, true);
            $ffi->train_linear_model($model, $this->datasetService->X_train, $this->datasetService->Y_train, $this->datasetService->data_size, 0.001, 10);
            
            // $testInput = FFI::new("float[2]", false, true);
            // foreach ($testInputs as $input) {
            //     $testInput[0] = $input[0];
            //     $testInput[1] = $input[1];
            //     $predictions[] = $ffi->predict_linear_model($model, $testInput);
            // }
            // foreach ($predictions as $prediction) {
            //     $this->info("Prediction: " . $prediction);
            // }

            // $ffi->free_linear_model($model);
            
        } catch (\Throwable $e) {
            $this->error("An error occurred: " . $e->getMessage());
            return 1;
        }

        $this->info('Model training completed successfully.');
        return 0;
    }
}
