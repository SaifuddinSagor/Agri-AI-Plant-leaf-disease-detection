<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AgriLeaf AI — Smart Leaf Disease Detector</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="{{ asset('css/style.css') }}" rel="stylesheet">
</head>
<body class="bg-light">
    <!-- Header -->
    <header class="bg-success text-white py-5">
        <div class="container">
            <h1 class="display-4">AgriLeaf AI</h1>
            <p class="lead">Empowering Farmers with AI-based Leaf Disease Detection</p>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container my-5">
        <!-- Unified Upload Section -->
        <div class="row justify-content-center">
            <div class="col-lg-8 col-md-10">
                <div class="card upload-card shadow-lg">
                    <div class="card-body p-5">
                        <div class="text-center mb-4">
                            <h2 class="display-5 mb-3">🌿 Leaf Disease Detection</h2>
                            <p class="lead text-muted">Upload a leaf image for instant AI-powered disease diagnosis and plant identification</p>
                        </div>
                        
                        <form id="uploadForm" enctype="multipart/form-data">
                            @csrf
                            
                            <!-- Image Upload -->
                            <div class="mb-4">
                                <label for="leafImage" class="form-label fw-bold">Upload Leaf Image</label>
                                <input type="file" class="form-control form-control-lg" id="leafImage" name="image" accept="image/*" required>
                                <div class="form-text">Supported formats: JPG, PNG, JPEG (Max: 2MB)</div>
                            </div>

                            <!-- Image Preview -->
                            <div id="imagePreview" class="mb-4 text-center d-none">
                                <p class="text-muted mb-2">Image Preview:</p>
                                <img src="" alt="Preview" class="img-fluid rounded shadow-sm">
                            </div>

                            <!-- Submit Button -->
                            <div class="d-grid">
                                <button type="submit" class="btn btn-success btn-lg">
                                    <i class="bi bi-cpu"></i> Analyze & Predict
                                </button>
                            </div>
                        </form>

                        <!-- Results Section -->
                        <div id="results" class="mt-5 d-none">
                            <hr class="my-4">
                            <h5 class="text-center mb-4">📊 Prediction Results</h5>
                            <div class="card result-card">
                                <div class="card-body">
                                    <div class="row">
                                        <div class="col-md-6 mb-3">
                                            <p class="mb-1 text-muted small">Plant Species</p>
                                            <p class="h5 text-primary mb-0"><span id="plantSpecies"></span></p>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <p class="mb-1 text-muted small">Disease Detected</p>
                                            <p class="h5 text-success mb-0"><span id="diseaseName"></span></p>
                                        </div>
                                        <div class="col-md-6 mb-3">
                                            <p class="mb-1 text-muted small">Confidence Level</p>
                                            <p class="h5 mb-0"><span id="confidence"></span>%</p>
                                        </div>
                                        <div class="col-12 mb-3">
                                            <p class="mb-1 text-muted small">Description</p>
                                            <p class="mb-0"><span id="description"></span></p>
                                        </div>
                                        <div class="col-12 mb-3">
                                            <p class="mb-1 text-muted small">Recommended Treatment</p>
                                            <p class="mb-0"><span id="cure"></span></p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="d-grid mt-4">
                                <button class="btn btn-outline-success btn-lg" onclick="resetUpload()">
                                    🔄 Analyze Another Image
                                </button>
                            </div>
                        </div>

                        <!-- Unknown Plant Error Section -->
                        <div id="unknownPlantError" class="mt-5 d-none">
                            <hr class="my-4">
                            <div class="card border-warning">
                                <div class="card-body text-center p-5">
                                    <div class="mb-4">
                                        <i class="bi bi-exclamation-triangle-fill text-warning" style="font-size: 3rem;"></i>
                                    </div>
                                    <h4 class="text-warning mb-3">🌿 Unknown Plant Detected</h4>
                                    <p class="lead text-muted mb-4" id="unknownPlantMessage">
                                        The uploaded image does not belong to any of the supported species — Hibiscus, Papaya, Bottle Gourd, or Tea.
                                        <br>Please upload a clear image of one of these plants' leaves for accurate disease prediction.
                                    </p>
                                    <div class="d-grid mt-4">
                                        <button class="btn btn-warning btn-lg" onclick="resetUpload()">
                                            🔄 Try Another Image
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Loading Spinner -->
                        <div id="loading" class="text-center mt-5 d-none">
                            <div class="spinner-border text-success" role="status" style="width: 3rem; height: 3rem;">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-3 h5 text-muted">🔍 Analyzing leaf image...</p>
                            <p class="text-muted small">This may take a few moments</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Footer -->
    <footer class="bg-dark text-white py-4 mt-5">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <p>&copy; 2025 AgriLeaf AI | Powered by Deep Learning</p>
                </div>
                <div class="col-md-6 text-md-end">
                    <a href="#" class="text-white text-decoration-none me-3">About</a>
                    <a href="#" class="text-white text-decoration-none">Contact</a>
                </div>
            </div>
        </div>
    </footer>

    <!-- Scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="{{ asset('js/script.js') }}"></script>
</body>
</html>