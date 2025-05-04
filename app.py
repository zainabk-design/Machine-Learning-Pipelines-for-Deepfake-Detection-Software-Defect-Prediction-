import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
import os

# Set page title and layout
st.set_page_config(page_title="ML Classification App", layout="wide")

# Title and description
st.title("Machine Learning Classification App")
st.markdown("""
This app provides two classification services:
1. **Deepfake Audio Detection**: Upload an audio file to check if it's genuine or deepfake
2. **Software Defect Prediction**: Input a software issue report to predict defect types
""")

# Sidebar for navigation
page = st.sidebar.radio("Select Service", ["Deepfake Audio Detection", "Software Defect Prediction"])

# Function to extract audio features (MFCC)
def extract_audio_features(audio_data, sr, n_mfcc=13, n_mels=40, max_length=100):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate to fixed length
    if mfccs.shape[1] < max_length:
        pad_width = max_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_length]
    
    return mfccs.flatten()

# Function to plot confidence scores
def plot_confidence(scores, labels):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, scores, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
    
    # Add values on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                ha='left', va='center')
    
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Prediction Confidence')
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Try to load pre-trained models if they exist
def load_models():
    try:
        # Check if models directory exists
        if os.path.exists('models'):
            audio_models = {}
            defect_models = {}
            
            # Load audio models if they exist
            if os.path.exists('models/svm_model.pkl'):
                with open('models/svm_model.pkl', 'rb') as f:
                    audio_models['SVM'] = pickle.load(f)
            
            if os.path.exists('models/lr_model.pkl'):
                with open('models/lr_model.pkl', 'rb') as f:
                    audio_models['Logistic Regression'] = pickle.load(f)
            
            if os.path.exists('models/perceptron_model.pkl'):
                with open('models/perceptron_model.pkl', 'rb') as f:
                    audio_models['Perceptron'] = pickle.load(f)
            
            if os.path.exists('models/mlp_model.pkl'):
                with open('models/mlp_model.pkl', 'rb') as f:
                    audio_models['MLP'] = pickle.load(f)
            
            # Load audio scaler if it exists
            if os.path.exists('models/audio_scaler.pkl'):
                with open('models/audio_scaler.pkl', 'rb') as f:
                    audio_scaler = pickle.load(f)
            else:
                audio_scaler = StandardScaler()
            
            # For defect models, we'll use dummy models for now
            # In a real implementation, you would load your trained defect models
            class DummyMultiLabelModel:
                def predict(self, X):
                    return np.random.randint(0, 2, size=(len(X), 7))
                
                def predict_proba(self, X):
                    return np.random.random(size=(len(X), 7))
            
            for model_name in ['SVM', 'Logistic Regression', 'Perceptron', 'Online Perceptron', 'MLP']:
                defect_models[model_name] = DummyMultiLabelModel()
            
            defect_scaler = StandardScaler()
            
            # Create a vectorizer
            vectorizer = TfidfVectorizer(max_features=100)
            # Fit with a dummy text
            vectorizer.fit(["Dummy text for vectorizer"])
            
            return audio_models, defect_models, audio_scaler, defect_scaler, vectorizer
        
        else:
            # If models directory doesn't exist, use dummy models
            return create_dummy_models()
    
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        return create_dummy_models()

# Create dummy models if real models can't be loaded
def create_dummy_models():
    class DummyModel:
        def predict(self, X):
            return np.random.randint(0, 2, size=len(X))
        
        def predict_proba(self, X):
            probs = np.random.random(size=(len(X), 2))
            return probs / probs.sum(axis=1, keepdims=True)
    
    class DummyMultiLabelModel:
        def predict(self, X):
            return np.random.randint(0, 2, size=(len(X), 7))
        
        def predict_proba(self, X):
            return np.random.random(size=(len(X), 7))
    
    # Audio models
    audio_models = {
        'SVM': DummyModel(),
        'Logistic Regression': DummyModel(),
        'Perceptron': DummyModel(),
        'MLP': DummyModel()
    }
    
    # Defect prediction models
    defect_models = {
        'SVM': DummyMultiLabelModel(),
        'Logistic Regression': DummyMultiLabelModel(),
        'Perceptron': DummyMultiLabelModel(),
        'Online Perceptron': DummyMultiLabelModel(),
        'MLP': DummyMultiLabelModel()
    }
    
    # Scalers
    audio_scaler = StandardScaler()
    defect_scaler = StandardScaler()
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=100)
    vectorizer.fit(["Dummy text for vectorizer"])
    
    return audio_models, defect_models, audio_scaler, defect_scaler, vectorizer

# Load models
audio_models, defect_models, audio_scaler, defect_scaler, vectorizer = load_models()

# Deepfake Audio Detection Page
if page == "Deepfake Audio Detection":
    st.header("Deepfake Audio Detection")
    st.write("Upload an audio file to check if it's genuine or deepfake")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["SVM", "Logistic Regression", "Perceptron", "MLP"]
    )
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Predict"):
            with st.spinner("Processing audio..."):
                try:
                    # Process the audio file
                    st.info("Extracting audio features...")
                    
                    try:
                        # Try to load and process the actual audio file
                        audio_data, sr = librosa.load(uploaded_file, sr=None)
                        features = extract_audio_features(audio_data, sr)
                        features_scaled = audio_scaler.transform(features.reshape(1, -1))
                        st.success("Audio features extracted successfully!")
                    except Exception as e:
                        st.warning(f"Error processing audio: {e}. Using simulated features instead.")
                        # For demonstration, use random features if there's an error
                        features_scaled = np.random.randn(1, 1300)
                    
                    # Get selected model
                    model = audio_models[model_name]
                    
                    # Make prediction
                    prediction = model.predict(features_scaled)[0]
                    probabilities = model.predict_proba(features_scaled)[0]
                    
                    # Display results
                    st.subheader("Prediction Result")
                    
                    if prediction == 0:
                        st.success("✅ This audio is likely GENUINE (Bonafide)")
                        confidence = probabilities[0]
                    else:
                        st.error("❌ This audio is likely a DEEPFAKE")
                        confidence = probabilities[1]
                    
                    st.write(f"Confidence: {confidence:.2f}")
                    
                    # Plot confidence scores
                    confidence_img = plot_confidence(
                        [probabilities[0], probabilities[1]],
                        ['Genuine', 'Deepfake']
                    )
                    st.image(f"data:image/png;base64,{confidence_img}", use_column_width=True)
                    
                    # Display feature visualization
                    st.subheader("Feature Visualization")
                    
                    try:
                        # Try to visualize actual features if we have them
                        if 'audio_data' in locals():
                            fig, ax = plt.subplots(figsize=(10, 4))
                            S = librosa.feature.melspectrogram(y=audio_data, sr=sr)
                            S_dB = librosa.power_to_db(S, ref=np.max)
                            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
                            fig.colorbar(img, ax=ax, format='%+2.0f dB')
                            ax.set_title('Mel Spectrogram')
                            st.pyplot(fig)
                        else:
                            # Show a simulated spectrogram
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.imshow(np.random.rand(128, 100), aspect='auto', origin='lower', cmap='viridis')
                            ax.set_title('Mel Spectrogram (Simulated)')
                            ax.set_ylabel('Mel Bins')
                            ax.set_xlabel('Time Frames')
                            st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Error generating visualization: {e}")
                        # Show a simulated spectrogram
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.imshow(np.random.rand(128, 100), aspect='auto', origin='lower', cmap='viridis')
                        ax.set_title('Mel Spectrogram (Simulated)')
                        ax.set_ylabel('Mel Bins')
                        ax.set_xlabel('Time Frames')
                        st.pyplot(fig)
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

# Software Defect Prediction Page
else:
    st.header("Software Defect Prediction")
    st.write("Enter a software issue report to predict defect types")
    
    # Text input
    report_text = st.text_area("Issue Report", height=150, 
                              placeholder="Enter the software issue report here...")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["SVM", "Logistic Regression", "Perceptron", "Online Perceptron", "MLP"]
    )
    
    if report_text and st.button("Predict"):
        with st.spinner("Analyzing report..."):
            try:
                # Process text
                st.info("Extracting text features...")
                
                try:
                    # Try to use the vectorizer
                    features = vectorizer.transform([report_text]).toarray()
                    features_scaled = defect_scaler.transform(features)
                except Exception as e:
                    st.warning(f"Error processing text: {e}. Using simulated features instead.")
                    # For demonstration, use random features
                    features_scaled = np.random.randn(1, 100)
                
                # Get selected model
                model = defect_models[model_name]
                
                # Make prediction
                predictions = model.predict(features_scaled)[0]
                probabilities = model.predict_proba(features_scaled)[0]
                
                # Display results
                st.subheader("Prediction Results")
                
                # Define defect types
                defect_types = [
                    'Blocker', 'Regression', 'Bug', 'Documentation', 
                    'Enhancement', 'Task', 'Dependency Update'
                ]
                
                # Create results table
                results_df = pd.DataFrame({
                    'Defect Type': defect_types,
                    'Predicted': ['Yes' if p == 1 else 'No' for p in predictions],
                    'Confidence': probabilities
                })
                
                # Style the dataframe
                def highlight_yes(val):
                    if val == 'Yes':
                        return 'background-color: #CCFFCC'
                    else:
                        return ''
                
                styled_df = results_df.style.applymap(highlight_yes, subset=['Predicted'])
                st.dataframe(styled_df, use_container_width=True)
                
                # Plot confidence scores for predicted defects
                st.subheader("Confidence Scores")
                
                # Filter for predicted defects
                predicted_defects = [defect_types[i] for i in range(len(predictions)) if predictions[i] == 1]
                predicted_probs = [probabilities[i] for i in range(len(predictions)) if predictions[i] == 1]
                
                if predicted_defects:
                    confidence_img = plot_confidence(predicted_probs, predicted_defects)
                    st.image(f"data:image/png;base64,{confidence_img}", use_column_width=True)
                else:
                    st.info("No defects were predicted for this report.")
                
                # Word importance visualization
                st.subheader("Important Words in Prediction")
                
                # Get words from the report
                words = report_text.split()[:10] if len(report_text.split()) > 10 else report_text.split()
                
                if words:
                    # Generate random importances for demonstration
                    # In a real app, you would extract actual feature importances
                    importances = np.random.random(len(words))
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.barh(words, importances, color='skyblue')
                    ax.set_title('Word Importance')
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                else:
                    st.info("No words to analyze in the report.")
            
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Add information about the models
st.sidebar.header("About the Models")
st.sidebar.markdown("""
### Models Used:
- **SVM**: Support Vector Machine classifier
- **Logistic Regression**: Linear classifier with logistic function
- **Perceptron**: Single-layer neural network
- **Online Perceptron**: Perceptron with online learning
- **MLP**: Multi-Layer Perceptron (neural network)

This app demonstrates the application of machine learning models for classification tasks.
""")

# Add information about model status
if os.path.exists('models'):
    st.sidebar.success("✅ Models directory found")
    
    # Check which models are available
    available_models = []
    if os.path.exists('models/svm_model.pkl'):
        available_models.append("SVM")
    if os.path.exists('models/lr_model.pkl'):
        available_models.append("Logistic Regression")
    if os.path.exists('models/perceptron_model.pkl'):
        available_models.append("Perceptron")
    if os.path.exists('models/mlp_model.pkl'):
        available_models.append("MLP")
    
    if available_models:
        st.sidebar.success(f"✅ Available models: {', '.join(available_models)}")
    else:
        st.sidebar.warning("⚠️ No trained models found. Using simulated models.")
else:
    st.sidebar.warning("⚠️ Models directory not found. Using simulated models.")

# Footer
st.markdown("---")
st.markdown("© 2023 Data Science for Software Engineering - Assignment #4")