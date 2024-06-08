<?php

use App\Http\Controllers\Process\ProcessController;
use Illuminate\Support\Facades\Route;

Route::get('/upload', function () {
    return view('image-classification');
})->name('upload.form');

Route::post('/upload', [ProcessController::class, 'upload'])->name('upload.image');
