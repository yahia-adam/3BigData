<?php


namespace App\Http\Process;

use Illuminate\View\Component;

class Process extends Component
{
    public function createLesson(){

        try{
            $path = $this->lesson['image']->store('lessons', 'images');

            //Create the product
            try{
                $stripeSecretKey = config('stripe.sk');
                $stripe = new \Stripe\StripeClient($stripeSecretKey);
                $product = $stripe->products->create([
                    'name' => $this->lesson['title'],
                    'active' => true,
                    'images' => [env('APP_URL') . $path]
                ]);

                //Create a subscription price
                $price = $stripe->prices->create([
                    'unit_amount' => (int)($this->lesson['price'] * 100),
                    'currency' => 'eur',
                    'product' => $product->id,
                ]);

                try{
                    $lesson = Lessons::create([
                        'title' => $this->lesson['title'],
                        'description' => $this->lesson['description'],
                        'image_url' => $path,
                        'difficulty' => $this->lesson['difficulty'],
                        'price' => $this->lesson['price'],
                        'created_at' => now(),
                        'updated_at' => now(),
                        'creator_id' => auth()->user()->id,
                        'product_id' => $product->id,
                        'price_id' => $price->id,
                    ]);
                    try{
                        foreach ($this->lesson_steps as $step){
                            $path = $step['video']->store('lessons', 'images');
                            LessonStep::create([
                                'sub_title' => $step['sub_title'],
                                'description' => $step['description'],
                                'order' => $step['order'],
                                'duration' => $step['duration'],
                                'video_url' => $path,
                                'lesson_id' => $lesson->id
                            ]);
                        }
                    }catch (Exception $e){
                        Log::error($e);
                        $this->errorMessage = "Couldn't create lesson's step in database.";
                    }
                }catch (Exception $e){
                    Log::error($e);
                    $this->errorMessage = "Couldn't create lesson in database.";
                }
            }catch(Exception $e){
                Log::error($e);
                $this->errorMessage = "Stripe api error occurred.";
            }
        }catch (\Exception $e){
            Log::error($e);
            $this->errorMessage = "An error occurred.";
        }
        $this->successMessage = "Lesson has been created successfully !";
    }

    public function addStep(){
        $this->number_lesson_steps++;
        $this->lesson_steps[$this->number_lesson_steps - 1]['order'] = $this->number_lesson_steps;
    }

    public function showStep(){
        dd($this->lesson, $this->lesson_steps);
    }

    public function update()
    {
        $this->validate([
            'lesson.image' => 'image|max:1024', // 1MB Max
            'lesson_steps.*.video' => 'image|max:1024', // 1MB Max
        ]);
    }
    public function render()
    {
        $lessons = Lessons::where('creator_id', auth()->user()->id)->get();
        return view('provider.lessons.create')->layout('layouts.provider');
    }
}
