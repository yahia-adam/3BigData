<?php

namespace App\Providers;
use Illuminate\Support\Facades\File;
use Illuminate\Support\ServiceProvider;
use Illuminate\Contracts\Foundation\Application;
use App\Services\DatasetService;

class DatasetServiceProvider extends ServiceProvider
{
    /**
     * Register services.
     */
    public function register(): void
    {
        $this->app->singleton(DatasetServiceProvider::class, function ($app) {
            return new DatasetService();
        });
    }

    /**
     * Bootstrap services.
     */
    public function boot(): void
    {

    }
}
