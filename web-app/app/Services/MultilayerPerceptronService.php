<?php

namespace App\Services;

use Illuminate\Support\Facades\File;
use Imagick;
use FFI;

class MultilayerPerceptronService
{
    protected $mylib;
    protected $model;
    protected $datasetService;

    public function __construct($datasetService)
    {
        $this->datasetService = $datasetService;
        $this->mylib = FFI::cdef("
            typedef struct MultiLayerPerceptron MultiLayerPerceptron;
            MultiLayerPerceptron* init_mlp(unsigned int* npl, unsigned int npl_size, bool is_classification);
            void train_mlp(MultiLayerPerceptron* model, float* inputs, float* outputs, unsigned int data_size, float alpha, unsigned int nb_iteration);
            float* predict_mlp(MultiLayerPerceptron* model, float* sample_inputs);
            const char* mlp_to_json(MultiLayerPerceptron* model);
            void save_mlp_model(MultiLayerPerceptron* model, const char* filepath);
            void free_mlp(MultiLayerPerceptron* model);
        ", env("LIB_PATH"));
    }

    public function trainNewModel()
    {
        $alpha = 0.01;
        $iterations = 1000;
        $npl = [27648, 128, 64, 3]; // Number of neurons per layer, adjust as necessary
        $npl_ptr = FFI::new("unsigned int[" . count($npl) . "]", false);
        foreach ($npl as $index => $value) {
            $npl_ptr[$index] = $value;
        }
        $this->model = $this->mylib->init_mlp(FFI::addr($npl_ptr[0]), count($npl), true);
        $this->mylib->train_mlp($this->model, FFI::cast('float*', $this->datasetService->X_train), FFI::cast('float*', $this->datasetService->Y_train), $this->datasetService->data_size, $alpha, $iterations);
       
        $ffi = FFI::cdef("
            void *memcpy(void *dest, const void *src, size_t n);
        ", "libc.so.6");
    
        $file_path = "/home/adam/esgi/pa/3BigData/web-app/storage/app/models/pmc/";
        $file = $ffi->new("char[" . (strlen($file_path) + 1) . "]", false);
        $ffi->memcpy($file, $file_path, strlen($file_path) + 1);
        dd(FFI::string($file));

        $this->mylib->save_mlp_model($this->model, $file);
    }

    public function predict($image_path)
    {
        $imageData = $this->datasetService->getImageData($image_path, $this->datasetService->image_width, $this->datasetService->image_height);
        $input = FFI::new("float[" . $this->datasetService->x_size . "]", false);
        foreach ($imageData as $index => $value) {
            $input[$index] = $value;
        }
        $result = $this->mylib->predict_mlp($this->model, FFI::cast('float*', $input));
        return FFI::array($result, $this->datasetService->y_size);
    }

    public function __destruct()
    {
        $this->mylib->free_mlp($this->model);
    }
}
