import json
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from typing import Dict, List, Tuple

class FaceMatcher:
    def __init__(self):
        # Initialize CLIP model and processor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load Pokémon features
        with open('pokemon_features.json', 'r', encoding='utf-8') as f:
            self.pokemon_features = json.load(f)
        
        # Load OpenCV face detection classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Define feature categories and their possible values
        self.feature_categories = {
            "face_shape": ["circular", "angular", "elongated", "triangular", "oval"],
            "eye_characteristics": ["large_anime", "small_anime", "slit", "round", "narrow"],
            "mouth_expression": ["wide_grin", "minimal", "neutral", "frowning", "smiling"],
            "nose_prominence": ["prominent", "minimal", "none", "pointed", "flat"],
            "dominant_colors": ["yellow", "red", "blue", "green", "purple", "brown", "white", "black", "pink", "orange"],
            "ear_horn_shape": ["cat_like", "horns", "none", "pointed", "rounded"]
        }
    
    def extract_face_features(self, image_path: str) -> Dict:
        try:
            # Read and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f"Could not read image: {image_path}")
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                raise Exception("No face detected in the image")
            
            # Get the first face detected
            (x, y, w, h) = faces[0]
            face_image = image[y:y+h, x:x+w]
            
            # Convert to RGB for CLIP
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(face_rgb)
            
            features = {}
            
            # Extract features for each category using CLIP
            for category, possible_values in self.feature_categories.items():
                # Prepare text inputs
                text_inputs = [f"a human face with {value} {category}" for value in possible_values]
                
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
            print(f"Error processing face image: {str(e)}")
            return None
    
    def calculate_similarity(self, face_features: Dict, pokemon_features: Dict) -> float:
        total_similarity = 0
        weights = {
            "face_shape": 0.25,
            "eye_characteristics": 0.20,
            "mouth_expression": 0.15,
            "nose_prominence": 0.15,
            "dominant_colors": 0.15,
            "ear_horn_shape": 0.10
        }
        
        for category, weight in weights.items():
            face_primary = face_features[category]["primary"]["feature"]
            face_confidence = face_features[category]["primary"]["confidence"]
            
            pokemon_primary = pokemon_features[category]["primary"]["feature"]
            pokemon_confidence = pokemon_features[category]["primary"]["confidence"]
            
            # Calculate similarity for this category
            if face_primary == pokemon_primary:
                similarity = 1.0
            else:
                # Check if primary matches secondary
                if face_primary == pokemon_features[category]["secondary"]["feature"]:
                    similarity = 0.7
                elif face_features[category]["secondary"]["feature"] == pokemon_primary:
                    similarity = 0.7
                else:
                    similarity = 0.3
            
            # Weight the similarity by confidence scores
            weighted_similarity = similarity * face_confidence * pokemon_confidence
            total_similarity += weighted_similarity * weight
        
        return total_similarity
    
    def find_matches(self, face_features: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
        matches = []
        
        for pokemon in self.pokemon_features:
            similarity = self.calculate_similarity(face_features, pokemon["features"])
            matches.append((pokemon["name"] + " " + pokemon["pokemon_id"], similarity))
        
        # Sort by similarity score and return top N matches
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]

def main():
    # Get user input for face image
    image_path = input("Enter the path to your face image: ")
    
    if not os.path.exists(image_path):
        print("Error: Image file not found!")
        return
    
    # Initialize matcher
    matcher = FaceMatcher()
    
    # Extract features from face
    print("Analyzing your face...")
    face_features = matcher.extract_face_features(image_path)
    
    if not face_features:
        print("Failed to analyze face features!")
        return
    
    # Find matches
    print("\nFinding matching Pokémon...")
    matches = matcher.find_matches(face_features)
    
    # Display results
    print("\nYour Pokémon matches:")
    print("-" * 40)
    for pokemon_name, similarity in matches:
        similarity_percentage = similarity * 100
        print(f"{pokemon_name}: {similarity_percentage:.1f}% match")

if __name__ == "__main__":
    main() 