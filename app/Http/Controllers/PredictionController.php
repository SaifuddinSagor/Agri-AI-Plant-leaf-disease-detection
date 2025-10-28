<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;

class PredictionController extends Controller
{
    public function index()
    {
        return view('home');
    }

    public function predict(Request $request)
    {
        try {
            $request->validate([
                'image' => 'required|image|max:2048'
            ]);
        } catch (\Illuminate\Validation\ValidationException $e) {
            return response()->json(['error' => 'Please select a valid image file.'], 422);
        }

        try {
            // Ensure storage directory exists
            $uploadPath = storage_path('app/public/uploads');
            if (!file_exists($uploadPath)) {
                mkdir($uploadPath, 0777, true);
            }

            // Store the uploaded image temporarily
            $imagePath = $request->file('image')->store('uploads', 'public');
            if (!$imagePath) {
                throw new \Exception('Failed to store the uploaded image.');
            }

            $fullImagePath = storage_path('app/public/' . $imagePath);
            if (!file_exists($fullImagePath)) {
                throw new \Exception('Stored image file not found.');
            }

            // Log file details for debugging
            \Log::debug('Upload details:', [
                'original_name' => $request->file('image')->getClientOriginalName(),
                'stored_path' => $imagePath,
                'full_path' => $fullImagePath,
                'exists' => file_exists($fullImagePath),
                'permissions' => substr(sprintf('%o', fileperms($fullImagePath)), -4)
            ]);

            // Execute the final Python script (no species parameter needed)
            $venvPythonPath = base_path('venv/bin/python3');
            $scriptPath = base_path('predict_combined_final.py');
            $command = "{$venvPythonPath} {$scriptPath} " . 
                      escapeshellarg($fullImagePath);
            
            // Debug info
            \Log::debug('Current directory: ' . getcwd());
            \Log::debug('Script exists: ' . (file_exists($scriptPath) ? 'yes' : 'no'));
            
            $output = shell_exec("/bin/bash -c " . escapeshellarg($command) . " 2>/dev/null"); // Only capture stdout, suppress stderr
            
            // Enhanced logging for debugging
            \Log::debug('Command executed: ' . $command);
            \Log::debug('Python script output: ' . $output);
            \Log::debug('Image path: ' . $fullImagePath);
            \Log::debug('Image exists: ' . (file_exists($fullImagePath) ? 'yes' : 'no'));
            
            // Delete the temporary image
            Storage::disk('public')->delete($imagePath);

            // Parse the JSON output from Python script
            $result = json_decode($output, true);

            if (is_null($result)) {
                return response()->json(['error' => 'Failed to parse prediction result. Please try again.'], 400);
            }

            if (isset($result['error'])) {
                return response()->json(['error' => $result['error']], 400);
            }

            return response()->json($result);

        } catch (\Exception $e) {
            return response()->json(['error' => $e->getMessage()], 500);
        }
    }
}