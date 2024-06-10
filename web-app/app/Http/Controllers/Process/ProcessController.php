<?php

namespace App\Http\Controllers\Process;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Intervention\Image\ImageManager as Image;
use Symfony\Component\VarDumper\Caster\FFICaster;

class ProcessController extends Controller
{
    /**
     * @throws \ImagickPixelIteratorException
     * @throws \ImagickException
     */
    public function upload(Request $request)
    {
        $request->validate([
            'image' => 'required|image|mimes:jpeg,png,jpg,gif|max:2048',
        ]);

        $imageFile = $request->file('image');
        $extension = $imageFile->getClientOriginalExtension();
        $filename = 'image.' . $extension;

        $imageData = base64_encode(file_get_contents($imageFile));
        $path = storage_path('app/public/images/' . $filename);
        $image = Image::imagick()->read($imageData);
        $image->resize(48, 48);
        $image->save($path);

        $publicPath = 'images/' . $filename;

        try {
            $imagick = new \Imagick($path);
            $matrix = [];
            $pixelIterator = $imagick->getPixelIterator();
            foreach ($pixelIterator as $pixels) {
                $row = [];
                foreach ($pixels as $pixel) {
                    $color = $pixel->getColor();
                    $row[] = $color;
                }
                $matrix[] = $row;
                $pixelIterator->syncIterator();
            }

            // dd($matrix);

            $action = $request->input('selectedAction');
            switch ($action) {
                case 'linearModelClassification':
                    $result = 'charbel';
                    break;
                case 'mlpClassification':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'rbfClassification':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'svmClassification':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'linearModelRegression':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'mlpRegression':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'rbfRegression':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                case 'kernelMethod':
                    // ici on appelle la fonction qui prend en parametre l'image et renvoie le type de poubelle pour afficher le resultat enfin
                    break;
                default:
                    return redirect()->route('upload.form')->withErrors(['message' => 'Action not known']);
            }

        } catch (\ImagickException $e) {
            return redirect()->route('upload.form')->withErrors(['message' => 'Error while transforming the image to matrix']);
        }

        return redirect()->route('upload.form')->with(['image' => $publicPath, 'result' => $result]);
    }

}
