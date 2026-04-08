"""
Download emotion speech samples from free public sources
"""
import os
import ssl
import urllib.request
import json
from pathlib import Path

# Disable SSL verification for downloading
ssl._create_default_https_context = ssl._create_unverified_context

data_dir = Path("../data/raw")
emotions_map = {
    "angry": "angry",
    "happy": "happy", 
    "neutral": "neutral",
    "sad": "sad"
}

# Create directories
for emotion in emotions_map.values():
    (data_dir / emotion).mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("EMOTION SPEECH DATASET DOWNLOADER")
print("=" * 60)

# Option 1: Manual dataset links you can download
print("\n📥 Option 1: Public Datasets (Free to download)")
print("-" * 60)
print("\n1. TESS Dataset (Toronto Emotional Speech Set)")
print("   URL: https://tspace.library.utoronto.ca/handle/1807/24612")
print("   Format: 2400 .wav files, 8 emotions")
print("   Size: ~50MB")
print("   Instructions:")
print("     - Download the .zip file")
print("     - Extract and filter files by emotion")
print("     - Copy to data/raw/{emotion}/ folders")

print("\n2. RAVDESS Dataset (Ryerson Audio-Visual Database)")
print("   URL: https://zenodo.org/record/1188976")
print("   Format: 1440 .wav files (speech only)")
print("   Size: ~285MB")

print("\n3. EmoDB Dataset (Berlin Database of Emotional Speech)")
print("   URL: http://www.emodb.bilderbar.info/")
print("   Format: 535 .wav files, 7 emotions")
print("   Size: ~170MB")

# Option 2: Create synthetic data locally
print("\n\n🎵 Option 2: Generate Test Data Locally")
print("-" * 60)
try:
    import librosa
    import numpy as np
    from scipy.io import wavfile
    
    print("\nGenerating synthetic audio data for testing...")
    
    # Generate simple synthetic audio for each emotion
    sample_rate = 22050
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    emotion_freqs = {
        "angry": [400, 500, 600],      # Higher frequencies for angry
        "happy": [300, 400, 500],      # Mid frequencies for happy
        "neutral": [200, 250, 300],    # Lower frequencies for neutral
        "sad": [150, 200, 250]         # Even lower for sad
    }
    
    for emotion, freqs in emotion_freqs.items():
        for i in range(5):  # 5 files per emotion
            # Create synthetic signal
            signal = np.zeros_like(t)
            for freq in freqs:
                signal += np.sin(2 * np.pi * freq * t) * (1 / len(freqs))
            
            # Add some variation
            signal = signal * (1 + 0.1 * np.random.randn(*signal.shape))
            signal = np.int16(signal * 32767 * 0.3)
            
            # Save file
            output_path = data_dir / emotion / f"synthetic_{i+1}.wav"
            wavfile.write(str(output_path), sample_rate, signal)
            print(f"  ✓ Created {output_path.name}")
    
    print("\n✅ Synthetic data created successfully!")
    print(f"   Location: {data_dir.resolve()}")
    print(f"   Files: 20 total (.wav files for testing)")
    print("\n   ⚠️  Note: These are synthetic test files only.")
    print("   For better results, download real emotion datasets listed above.")
    
except ImportError:
    print("\n❌ librosa/scipy not installed")
    print("   Run: pip install librosa scipy")

print("\n" + "=" * 60)
print("After downloading/generating data:")
print("1. Extract/organize audio into data/raw/ folders")
print("2. Run: python generate_features.py")
print("3. Run: python train.py")
print("=" * 60)
