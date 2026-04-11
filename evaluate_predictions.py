import os
import random
from src.predict import predict

class DummyFile:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.data = f.read()
    def getvalue(self):
        return self.data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

classes = ["angry", "happy", "neutral", "sad"]
total_correct = 0
total_files = 0

print("Starting Automated Evaluation...\n")

for emotion in classes:
    class_dir = os.path.join(RAW_DIR, emotion)
    
    # Grab 10 random files from the emotion folder
    all_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
    sample_files = random.sample(all_files, min(len(all_files), 15))
    
    correct = 0
    
    print(f"Testing Class: {emotion.upper()}")
    for file in sample_files:
        path = os.path.join(class_dir, file)
        
        # Simulate Streamlit interface
        stream_file_obj = DummyFile(path)
        prediction = predict(stream_file_obj)
        
        if prediction == emotion:
            correct += 1
        else:
            print(f"  [X] Failed: {file} -> Predicted: {prediction}")
            
    total_correct += correct
    total_files += len(sample_files)
    print(f"  => Accuracy for {emotion}: {correct}/{len(sample_files)} ({(correct/len(sample_files))*100:.1f}%)\n")

print("="*40)
print(f"Overall Simulated UI Accuracy: {total_correct}/{total_files} ({(total_correct/total_files)*100:.1f}%)")
print("="*40)
