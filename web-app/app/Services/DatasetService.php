<?php

namespace App\Services;

use Illuminate\Support\Facades\File;
use Imagick;
use FFI;

class DatasetService
{
    public $dataset_path;
    public $images;
    public $X_train;
    public $Y_train;
    public $X_test;
    public $Y_test;
    public $x_size;
    public $y_size;
    public $data_size;
    public $image_width;
    public $image_height;

    public function __construct()
    {
        $this->dataset_path = env('DATASET_PATH', "../datasets");
        $this->init_dataset();
    }

    public function getDataset($type, $category)
    {
        $path = $this->dataset_path . "/{$category}/{$type}";
        $files = collect(File::allFiles($path));

        return $files->map(function ($file) {
            return $file->getPathname();
        });
    }

    public function getImageData($path, $width, $height)
    {
        $image = new Imagick($path);
        return $image->exportImagePixels(0, 0, $width, $height, "RGB", Imagick::PIXEL_CHAR);
    }

    public function init_dataset()
    {
        $paper_train = $this->getDataset('train', 'paper');
        $plastic_train = $this->getDataset('train', 'plastic');
        $glass_train = $this->getDataset('train', 'glass');
        
        $this->data_size = count($paper_train) + count($plastic_train) + count($glass_train);
        $this->image_height = 48;
        $this->image_width = 48;
        $this->x_size = $this->image_height * $this->image_width * 3; // 3 for RGB
        $this->y_size = 3; // Assuming 3 classes
    
        $X_train_type = FFI::arrayType(FFI::type("float"), [$this->data_size * $this->x_size]);
        $Y_train_type = FFI::arrayType(FFI::type("float"), [$this->data_size * $this->y_size]);
    
        $this->X_train = FFI::new($X_train_type);
        $this->Y_train = FFI::new($Y_train_type);
    
        $xIndex = 0;
        $yIndex = 0;
        $classes = ['paper' => [1, 0, 0], 'plastic' => [0, 1, 0], 'glass' => [0, 0, 1]];
    
        foreach (['paper_train' => $paper_train, 'plastic_train' => $plastic_train, 'glass_train' => $glass_train] as $label => $data_set) {
            foreach ($data_set as $data) {
                $imageData = $this->getImageData($data, $this->image_width, $this->image_height);
                foreach ($imageData as $d) {
                    $this->X_train[$xIndex++] = $d / 255.0; // Normalization to 0-1 range
                }
                foreach ($classes[str_replace('_train', '', $label)] as $value) {
                    $this->Y_train[$yIndex++] = $value;
                }
            }
        }
    }
    
}
