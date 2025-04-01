import json
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from typing import Dict, List
import time

class PokemonFeatureExtractor:
    def __init__(self):
        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Define feature categories and their possible values
        self.feature_categories = {
            "face_shape": ["circular", "angular", "elongated", "triangular", "oval"],
            "eye_characteristics": ["large_anime", "small_anime", "slit", "round", "narrow"],
            "mouth_expression": ["wide_grin", "minimal", "neutral", "frowning", "smiling"],
            "nose_prominence": ["prominent", "minimal", "none", "pointed", "flat"],
            "dominant_colors": ["yellow", "red", "blue", "green", "purple", "brown", "white", "black", "pink", "orange"],
            "ear_horn_shape": ["cat_like", "horns", "none", "pointed", "rounded"]
        }
        
    def extract_features(self, image_path: str) -> Dict:
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Could not read image: {image_path}")
            
            # Convert to RGB for CLIP
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            features = {}
            
            # Extract features for each category using CLIP
            for category, possible_values in self.feature_categories.items():
                # Prepare text inputs
                text_inputs = [f"a pokemon with {value} {category}" for value in possible_values]
                
                # Process inputs
                inputs = self.processor(
                    images=pil_image,
                    text=text_inputs,
                    return_tensors="pt",
                    padding=True
                )
                
                # Get predictions
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Get top 2 features for each category
                top2_probs, top2_indices = torch.topk(probs, 2)
                
                features[category] = {
                    "primary": {
                        "feature": possible_values[top2_indices[0][0]],
                        "confidence": float(top2_probs[0][0])
                    },
                    "secondary": {
                        "feature": possible_values[top2_indices[0][1]],
                        "confidence": float(top2_probs[0][1])
                    }
                }
            
            return features
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

def main():
    # Load Pokémon data
    with open('pokemon_data.json', 'r', encoding='utf-8') as f:
        pokemon_data = json.load(f)
    
    # Initialize feature extractor
    extractor = PokemonFeatureExtractor()
    
    # Process each Pokémon
    features_data = []
    
    for pokemon in pokemon_data:
        if not pokemon.get('image_path') or not os.path.exists(pokemon['image_path']):
            print(f"Skipping {pokemon['name']} - image not found")
            continue
            
        print(f"Processing {pokemon['name']}...")
        features = extractor.extract_features(pokemon['image_path'])
        
        if features:
            features_data.append({
                'pokemon_id': pokemon['number'],
                'name': pokemon['name'],
                'features': features
            })
        
        # Add a small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    # Save features to JSON
    with open('pokemon_features.json', 'w', encoding='utf-8') as f:
        json.dump(features_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processed {len(features_data)} Pokémon")
    print("Features saved to pokemon_features.json")

if __name__ == "__main__":
    main() 