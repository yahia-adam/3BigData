<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use App\Services\DatasetService;
use App\Services\MultilayerPerceptronService;

class AppServiceProvider extends ServiceProvider
{
    
    /**
     * Register any application services.
     */
    public function register(): void
    {
        $this->app->singleton(DatasetService::class, function ($app) {
            return new DatasetService();
        });
    
        $this->app->singleton(MultilayerPerceptronService::class, function ($app) {
            $datasetService = $app->make(DatasetService::class);
            return new MultilayerPerceptronService($datasetService);
        });
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        //
    }
}
