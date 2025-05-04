import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier  # Using sklearn's MLP instead of TensorFlow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import librosa
import warnings
warnings.filterwarnings('ignore')

# For demonstration purposes, we'll simulate loading the dataset
# In a real implementation, you would use:
# from datasets import load_dataset
# ds = load_dataset("CSALT/deepfake_detection_dataset_urdu")

# Simulating dataset for demonstration
class SimulatedDataset:
    def __init__(self):
        # Create a simulated dataset with 200 samples
        np.random.seed(42)
        self.audio_paths = [f"audio_{i}.wav" for i in range(200)]
        self.labels = np.random.choice([0, 1], size=200)  # 0: Bonafide, 1: Deepfake
        
    def __getitem__(self, idx):
        return {"audio_path": self.audio_paths[idx], "label": self.labels[idx]}
    
    def __len__(self):
        return len(self.audio_paths)

# Feature extraction function
def extract_audio_features(audio_path, n_mfcc=13, n_mels=40, max_length=100):
    """
    Extract MFCC features from audio file
    In a real implementation, you would load the actual audio file
    """
    # Simulate feature extraction
    # In a real implementation:
    # y, sr = librosa.load(audio_path, sr=None)
    # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # For demonstration, generate random features
    mfccs = np.random.randn(n_mfcc, max_length)
    
    # Pad or truncate to fixed length
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    return mfccs.flatten()  # Flatten to 1D array

def main():
    # Load dataset
    print("Loading dataset...")
    ds = SimulatedDataset()

    # Extract features and prepare data
    print("Extracting features...")
    X = []
    y = []

    for i in range(len(ds)):
        sample = ds[i]
        features = extract_audio_features(sample["audio_path"])
        X.append(features)
        y.append(sample["label"])

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model evaluation function
    def evaluate_model(y_true, y_pred, y_prob=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if y_prob is not None:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = roc_auc_score(y_true, y_pred)
        
        return {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'AUC-ROC': auc
        }

    # 1. Support Vector Machine (SVM)
    print("\nTraining SVM model...")
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    svm_prob = svm_model.predict_proba(X_test_scaled)[:, 1]
    svm_metrics = evaluate_model(y_test, svm_pred, svm_prob)

    # 2. Logistic Regression
    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_metrics = evaluate_model(y_test, lr_pred, lr_prob)

    # 3. Single-Layer Perceptron
    print("Training Perceptron model...")
    perceptron_model = Perceptron(max_iter=1000)
    perceptron_model.fit(X_train_scaled, y_train)
    perceptron_pred = perceptron_model.predict(X_test_scaled)
    perceptron_metrics = evaluate_model(y_test, perceptron_pred)

    # 4. Multi-Layer Perceptron (MLP) - Using sklearn's MLP instead of TensorFlow
    print("Training MLP model...")
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', 
                             max_iter=300, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    mlp_pred = mlp_model.predict(X_test_scaled)
    mlp_prob = mlp_model.predict_proba(X_test_scaled)[:, 1]
    mlp_metrics = evaluate_model(y_test, mlp_pred, mlp_prob)

    # Compare model performance
    print("\nModel Performance Comparison:")
    models = ['SVM', 'Logistic Regression', 'Perceptron', 'MLP']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    results = pd.DataFrame({
        'SVM': [svm_metrics[m] for m in metrics],
        'Logistic Regression': [lr_metrics[m] for m in metrics],
        'Perceptron': [perceptron_metrics[m] for m in metrics],
        'MLP': [mlp_metrics[m] for m in metrics]
    }, index=metrics)

    print(results)

    # Visualize results
    plt.figure(figsize=(12, 8))
    results.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
    print("Model performance chart saved as 'model_performance.png'")

    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (model_name, y_pred) in enumerate([
        ('SVM', svm_pred),
        ('Logistic Regression', lr_pred),
        ('Perceptron', perceptron_pred),
        ('MLP', mlp_pred)
    ]):
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bonafide', 'Deepfake'])
        disp.plot(ax=axes[i], cmap='Blues', values_format='d')
        axes[i].set_title(f'{model_name} Confusion Matrix')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()
    print("Confusion matrices saved as 'confusion_matrices.png'")

    # Save models for later use
    import pickle
    import os
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save models
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    
    with open('models/lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    
    with open('models/perceptron_model.pkl', 'wb') as f:
        pickle.dump(perceptron_model, f)
    
    with open('models/mlp_model.pkl', 'wb') as f:
        pickle.dump(mlp_model, f)
    
    # Save scaler
    with open('models/audio_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModels saved to 'models/' directory")
    print("\nDeepfake Audio Detection Pipeline Complete!")

if __name__ == "__main__":
    main()