"""
🤖 ENERGY PREDICTOR - WITH REAL VIDEO FEATURE EXTRACTION
ML-powered energy optimization using actual video features
"""

import joblib
import numpy as np
import os

# Try to import video feature extractor
try:
    from video_feature_extractor import extractor as video_extractor
    VIDEO_EXTRACTION_AVAILABLE = True
except ImportError:
    VIDEO_EXTRACTION_AVAILABLE = False
    print("⚠️  video_feature_extractor not found - using default features")

class EnergyPredictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.device_encoder = None
        self.feature_columns = None
        self.is_loaded = False
        self.load_models()
    
    def load_models(self):
        try:
            # Try loading files with different names
            model_files = ['energy_optimizer_model.pkl', 'energy_optimizer_model (1).pkl']
            encoder_files = ['device_encoder.pkl', 'device_encoder (1).pkl']
            feature_files = ['feature_columns.pkl', 'feature_columns (1).pkl']
            
            for f in model_files:
                path = os.path.join(self.model_dir, f)
                if os.path.exists(path):
                    self.model = joblib.load(path)
                    break
            
            for f in encoder_files:
                path = os.path.join(self.model_dir, f)
                if os.path.exists(path):
                    self.device_encoder = joblib.load(path)
                    break
            
            for f in feature_files:
                path = os.path.join(self.model_dir, f)
                if os.path.exists(path):
                    self.feature_columns = joblib.load(path)
                    break
            
            if self.model and self.device_encoder and self.feature_columns:
                self.is_loaded = True
                print("✅ ML models loaded successfully!")
                print(f"   Device types: {list(self.device_encoder.classes_)}")
                if VIDEO_EXTRACTION_AVAILABLE:
                    print("✅ Video feature extraction: ENABLED")
                else:
                    print("⚠️  Video feature extraction: DISABLED (using defaults)")
                return True
            else:
                print("❌ Could not load all model files")
                return False
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_power_from_video(self, video_path, device_type='BrandA-Model1', 
                                 luminance_reduction=0):
        """
        Predict power consumption by extracting features from actual video file.
        
        Args:
            video_path (str): Path to video file
            device_type (str): Device model name
            luminance_reduction (int): Brightness reduction % (0-100)
        
        Returns:
            dict: {
                'predicted_power_mw': float,
                'video_features': dict,
                'using_real_features': bool
            }
        """
        if not self.is_loaded:
            return None
        
        # Extract real video features
        if VIDEO_EXTRACTION_AVAILABLE and os.path.exists(video_path):
            video_features = video_extractor.extract_for_ml_model(
                video_path, device_type, luminance_reduction
            )
            using_real_features = True
        else:
            # Fallback to defaults
            video_features = {
                'device_type': device_type,
                'bitrate': 3000,
                'luminance_reduction': luminance_reduction,
                'resolution': 1080,
                'duration': 10,
                'luminance': 80
            }
            using_real_features = False
        
        # Predict power
        power = self.predict_power(
            device_type=video_features['device_type'],
            bitrate=video_features['bitrate'],
            luminance_reduction=video_features['luminance_reduction'],
            resolution=video_features['resolution'],
            duration=video_features['duration'],
            luminance=video_features['luminance']
        )
        
        return {
            'predicted_power_mw': power,
            'video_features': video_features,
            'using_real_features': using_real_features
        }
    
    def predict_power(self, device_type, bitrate, luminance_reduction, 
                     resolution, duration=10, luminance=80):
        """Predict power consumption for given configuration."""
        if not self.is_loaded:
            return None
        
        try:
            if device_type in self.device_encoder.classes_:
                device_encoded = self.device_encoder.transform([device_type])[0]
            else:
                device_encoded = 0
            
            features = [[device_encoded, bitrate, luminance_reduction, 
                        resolution, duration, luminance]]
            power = self.model.predict(features)[0]
            return float(power)
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def recommend_settings(self, battery_level, bandwidth_kbps, 
                          device_type='BrandA-Model1', video_path=None):
        """
        Recommend optimal video quality and brightness.
        
        Args:
            battery_level (int): Current battery % (0-100)
            bandwidth_kbps (int): Available bandwidth in kbps
            device_type (str): Device model
            video_path (str, optional): Path to video for feature extraction
        
        Returns:
            dict: Recommended settings with predicted power
        """
        if not self.is_loaded:
            return {'error': 'ML model not loaded', 'quality': '720p', 'brightness': 100}
        
        # Extract video features if path provided
        video_luminance = 80  # Default
        if video_path and VIDEO_EXTRACTION_AVAILABLE and os.path.exists(video_path):
            features = video_extractor.extract_features(video_path)
            video_luminance = features['luminance']
        
        qualities = [
            {'name': '1080p', 'resolution': 1080, 'bitrate': 3000},
            {'name': '720p', 'resolution': 720, 'bitrate': 1800},
            {'name': '480p', 'resolution': 480, 'bitrate': 900},
            {'name': '360p', 'resolution': 360, 'bitrate': 600},
        ]
        
        # Battery-aware luminance reduction
        if battery_level > 80:
            luminance_reduction = 10
        elif battery_level > 60:
            luminance_reduction = 20
        elif battery_level > 40:
            luminance_reduction = 30
        elif battery_level > 20:
            luminance_reduction = 40
        else:
            luminance_reduction = 50
        
        brightness_percent = 100 - luminance_reduction
        
        candidates = []
        for quality in qualities:
            if quality['bitrate'] <= bandwidth_kbps * 1.2:
                power = self.predict_power(
                    device_type, quality['bitrate'], luminance_reduction, 
                    quality['resolution'], luminance=video_luminance
                )
                
                if power is not None:
                    if battery_level > 70:
                        score = (quality['resolution'] / 1080 * 0.6) + ((100 - power) / 100 * 0.4)
                    elif battery_level > 40:
                        score = (quality['resolution'] / 1080 * 0.5) + ((100 - power) / 100 * 0.5)
                    else:
                        score = (quality['resolution'] / 1080 * 0.3) + ((100 - power) / 100 * 0.7)
                    
                    candidates.append({
                        'quality': quality['name'],
                        'resolution': quality['resolution'],
                        'bitrate': quality['bitrate'],
                        'power': power,
                        'score': score
                    })
        
        if candidates:
            best = max(candidates, key=lambda x: x['score'])
            return {
                'quality': best['quality'],
                'brightness': brightness_percent,
                'predicted_power_mw': round(best['power'], 2),
                'battery_level': battery_level,
                'video_luminance': video_luminance
            }
        else:
            return {'quality': '360p', 'brightness': brightness_percent, 'battery_level': battery_level}

predictor = EnergyPredictor()

if __name__ == '__main__':
    print("="*60)
    print("🧪 TESTING ENERGY PREDICTOR")
    print("="*60)
    
    if predictor.is_loaded:
        print("\n1️⃣ Testing power prediction:")
        power = predictor.predict_power('BrandA-Model1', 3000, 10, 1080)
        print(f"   Predicted power: {power:.2f} mW")
        
        print("\n2️⃣ Testing AI recommendation:")
        rec = predictor.recommend_settings(60, 3000)
        print(f"   Quality: {rec['quality']}")
        print(f"   Brightness: {rec['brightness']}%")
        print(f"   Power: {rec['predicted_power_mw']} mW")
        
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Models not loaded!")
    
    print("="*60)