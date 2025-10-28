import sys
import json
import os
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from keras.utils import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Compiled the loaded model')

# Enable unsafe deserialization for lambda layers
keras.config.enable_unsafe_deserialization()

class AddPositionEmbs(keras.layers.Layer):
    """Adds learnable positional embeddings to the inputs."""

    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.posemb = None

    def build(self, input_shape):
        self.posemb = self.add_weight(
            name='pos_embedding',
            shape=[1, self.seq_len, self.embed_dim],
            initializer='zeros')

    def call(self, inputs):
        return inputs + self.posemb

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'embed_dim': self.embed_dim
        })
        return config

def add_cls_token(inputs):
    """Add a CLS token to the input tensor."""
    batch_size = tf.shape(inputs)[0]
    cls_init = tf.zeros([batch_size, 1, inputs.shape[-1]])
    return tf.concat([cls_init, inputs], axis=1)

def get_cls(inputs):
    """Extract the CLS token from a sequence tensor (assumes CLS at index 0)."""
    return inputs[:, 0]

def load_and_preprocess_image(image_path):
    """Load and preprocess the image for model prediction."""
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_class_mapping(species):
    """Get the class mapping for a specific species from the JSON file."""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public/assets/data/agriculture_datasets.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    dataset = next((d for d in data['datasets'] if d['species'].lower() == species.lower()), None)
    if not dataset:
        raise ValueError(f'Species {species} not found in dataset')
    
    return dataset['classes']

def get_all_species():
    """Get all available species from the JSON file."""
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'public/assets/data/agriculture_datasets.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return [d['species'] for d in data['datasets']]

def load_model():
    """Load the saved model."""
    import tensorflow as tf
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public/models/combined_saved_model")
    
    try:
        model = tf.saved_model.load(model_path)
        return model
    except Exception as e:
        raise Exception(f"Failed to load saved model: {e}")

def get_class_ranges():
    """Get the class ranges for each species in the combined model."""
    # Based on testing, the model seems to be trained with a different order
    # Let's try: Tea, Hibiscus, Bottle Gourd, Papaya
    class_ranges = {
        'Hibiscus': {
            'start': 0,
            'end': 7,
            'classes': [
                {'name': 'Healthy', 'cure': 'No treatment needed. Maintain regular watering and monitor for early signs of disease.', 'description': 'Leaves without visible damage or disease.'},
                {'name': 'Mild Edge Damage', 'cure': 'Trim affected areas and apply protective fungicide.', 'description': 'Minor damage along leaf edges, often from environmental stress.'},
                {'name': 'Citruspot', 'cure': 'Apply copper-based fungicide and improve air circulation.', 'description': 'Circular spots resembling citrus canker symptoms.'},
                {'name': 'Slightly Diseased', 'cure': 'Remove affected leaves and apply systemic fungicide.', 'description': 'Early stage disease with minor visible symptoms.'},
                {'name': 'Early Mild Spotting', 'cure': 'Use Mancozeb 75% WP fungicide and avoid overhead watering.', 'description': 'Tiny, scattered dark spots at early infection stage.'},
                {'name': 'Wrinkled', 'cure': 'Check for viral infection and apply appropriate treatment.', 'description': 'Leaves showing wrinkled or distorted appearance.'},
                {'name': 'Senescent', 'cure': 'Natural aging process, no treatment needed.', 'description': 'Older leaves showing natural aging signs.'},
                {'name': 'Fungal Infected', 'cure': 'Apply broad-spectrum fungicide and improve plant hygiene.', 'description': 'Clear fungal infection with visible spores or lesions.'}
            ]
        },
        'Tea': {
            'start': 8,
            'end': 12,
            'classes': [
                {'name': 'Algal Leaf Spot', 'cure': 'Apply copper-based fungicide and improve air circulation.', 'description': 'Circular spots with greenish appearance.'},
                {'name': 'Brown Blight', 'cure': 'Remove affected leaves and apply fungicide treatment.', 'description': 'Brown lesions on leaves causing tissue death.'},
                {'name': 'Grey Blight', 'cure': 'Use systemic fungicides and maintain proper spacing.', 'description': 'Greyish spots that spread across leaf surface.'},
                {'name': 'Healthy', 'cure': 'Maintain balanced fertilizer and regular pest inspection.', 'description': 'Leaves without disease symptoms.'},
                {'name': 'Red Leaf Spot', 'cure': 'Apply appropriate fungicide and improve drainage.', 'description': 'Reddish spots indicating fungal infection.'}
            ]
        },
        'Bottle Gourd': {
            'start': 13,
            'end': 20,
            'classes': [
                {'name': 'Anthracnose', 'cure': 'Use copper-based fungicides and remove infected debris.', 'description': 'Sunken dark spots on leaves caused by Colletotrichum fungus.'},
                {'name': 'Mosaic Virus', 'cure': 'Remove infected plants and control aphid vectors.', 'description': 'Mottled yellow and green patterns on leaves.'},
                {'name': 'Alternaria Leaf Blight', 'cure': 'Apply fungicide and improve air circulation.', 'description': 'Dark brown spots with concentric rings.'},
                {'name': 'Healthy', 'cure': 'Maintain proper watering and pest control.', 'description': 'Leaves without any visible disease symptoms.'},
                {'name': 'Downy Mildew', 'cure': 'Use systemic fungicides and improve drainage.', 'description': 'Yellow spots on upper surface with white mold underneath.'},
                {'name': 'Early Alternaria Leaf Blight', 'cure': 'Apply preventive fungicide treatment.', 'description': 'Initial stage of Alternaria infection with small spots.'},
                {'name': 'Fungal Damage Leaf', 'cure': 'Remove affected leaves and apply fungicide.', 'description': 'General fungal damage with visible lesions.'},
                {'name': 'Angular Leaf Spot', 'cure': 'Use copper-based treatment and improve spacing.', 'description': 'Angular-shaped spots limited by leaf veins.'}
            ]
        },
        'Papaya': {
            'start': 21,
            'end': 26,
            'classes': [
                {'name': 'Bacterial Blight', 'cure': 'Apply copper-based bactericide and remove infected parts.', 'description': 'Water-soaked lesions that turn brown and dry out.'},
                {'name': 'Carica Insect Hole', 'cure': 'Use appropriate insecticide and maintain plant hygiene.', 'description': 'Holes in leaves caused by insect feeding damage.'},
                {'name': 'Curled Yellow Spot', 'cure': 'Check for viral infection and control vectors.', 'description': 'Yellow spots with curled leaf edges.'},
                {'name': 'Healthy Leaf', 'cure': 'Regular watering and pest control.', 'description': 'Leaves without any visible disease symptoms.'},
                {'name': 'Pathogen Symptoms', 'cure': 'Identify specific pathogen and apply targeted treatment.', 'description': 'General symptoms indicating pathogen presence.'},
                {'name': 'Yellow Necrotic Spots Holes', 'cure': 'Remove affected leaves and apply appropriate fungicide.', 'description': 'Yellow spots that later develop into necrosis, often from viral infections.'}
            ]
        }
    }
    
    return class_ranges

def predict_disease_auto(image_path):
    """Predict the disease for a given leaf image using the combined model with automatic species detection."""
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        processed_image = load_and_preprocess_image(image_path)
        
        # Convert numpy array to TensorFlow tensor
        import tensorflow as tf
        processed_tensor = tf.constant(processed_image, dtype=tf.float32)
        
        # Make prediction using the saved model
        # Try different ways to call the model
        try:
            predictions = model.signatures['serving_default'](input_layer_1=processed_tensor)
            if hasattr(predictions, 'values'):
                predictions = list(predictions.values())[0]
        except:
            try:
                predictions = model(input_layer_1=processed_tensor)
            except:
                # Try to get the inference function
                inference_func = model.signatures['serving_default']
                predictions = inference_func(input_layer_1=processed_tensor)
                if hasattr(predictions, 'values'):
                    predictions = list(predictions.values())[0]
        
        # Convert to numpy if needed
        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = int(np.argmax(predictions[0]))
        
        # Get class ranges for all species
        class_ranges = get_class_ranges()
        
        # Find which species the predicted class belongs to
        best_species = None
        best_disease = None
        relative_class = None
        
        for species, range_info in class_ranges.items():
            if range_info['start'] <= predicted_class <= range_info['end']:
                best_species = species
                relative_class = predicted_class - range_info['start']
                best_disease = range_info['classes'][relative_class]
                break
        
        # Check if confidence is too low (likely not a supported plant)
        if confidence < 10.0:  # Lower threshold for unknown plants
            return json.dumps({
                'error': 'unknown_plant',
                'message': 'The uploaded image does not belong to any of the supported species — Hibiscus, Papaya, Bottle Gourd, or Tea. Please upload a clear image of one of these plants\' leaves for accurate disease prediction.'
            })
        
        if not best_species or not best_disease:
            return json.dumps({
                'error': 'unknown_plant',
                'message': 'The uploaded image does not belong to any of the supported species — Hibiscus, Papaya, Bottle Gourd, or Tea. Please upload a clear image of one of these plants\' leaves for accurate disease prediction.'
            })
        
        # Return results
        result = {
            'species': best_species,
            'disease': best_disease['name'],
            'confidence': confidence,
            'cure': best_disease['cure'],
            'description': best_disease.get('description', ''),
            'model_type': 'SavedModel'
        }
        return json.dumps(result)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        return json.dumps({'error': error_msg})

def predict_disease(image_path, species):
    """Predict the disease for a given leaf image using the combined model."""
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        processed_image = load_and_preprocess_image(image_path)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        confidence = float(np.max(predictions[0]) * 100)
        predicted_class = int(np.argmax(predictions[0]))
        
        # Get class mapping for the species
        classes = get_class_mapping(species)
        
        # Check if predicted class is valid
        if predicted_class >= len(classes):
            error_msg = f"Model predicted class {predicted_class} but we only have {len(classes)} classes for {species}"
            return json.dumps({'error': error_msg})
        
        disease = classes[predicted_class]
        
        # Return results
        result = {
            'species': species,
            'disease': disease['name'],
            'confidence': confidence,
            'cure': disease['cure'],
            'description': disease.get('description', ''),
            'model_type': 'H5'
        }
        return json.dumps(result)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        return json.dumps({'error': error_msg})

if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Auto-detect species mode
        image_path = sys.argv[1]
        print(predict_disease_auto(image_path))
    elif len(sys.argv) == 3:
        # Manual species mode
        image_path = sys.argv[1]
        species = sys.argv[2]
        print(predict_disease(image_path, species))
    else:
        print(json.dumps({'error': 'Invalid arguments. Usage: python script.py <image_path> [species]'}))
        sys.exit(1)
