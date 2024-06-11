<?php

namespace App\Services;

use Illuminate\Support\Facades\File;
use Imagick;
use FFI;

class DatasetService
{

    protected $dataset_service;
    protected $mylib;

    protected $model;

    public function __construct()
    {
        $this->dataset_service = new DatasetService();
        $this->mylib = FFI::cdef("
            typedef struct MultiLayerPerceptron MultiLayerPerceptron;
            MultiLayerPerceptron* init_mlp(unsigned int* npl, unsigned int npl_size, bool is_classification);
            void train_mlp(
                MultiLayerPerceptron* model, 
                float* inputs, 
                float* outputs, 
                unsigned int data_size, 
                float alpha, 
                unsigned int nb_iteration);
            float* predict_mlp(MultiLayerPerceptron* model, float* sample_inputs);
            const char* mlp_to_json(MultiLayerPerceptron* model);
            void save_mlp_model(MultiLayerPerceptron* model, const char* filepath);
            void free_mlp(MultiLayerPerceptron* model);
        ", env("LIB_PATH"));
    }

    public function train_new_model()
    {
        
    }
    public function load_model()
    {

    }

    public function predict($image)
    {

    }
}
