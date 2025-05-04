import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import hamming_loss, f1_score, precision_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# For demonstration, create a simulated dataset based on the image provided
def create_simulated_dataset():
    # Create sample data similar to what was shown in the image
    reports = [
        "The mention of Fix S",
        "It seems like you need",
        "The issue AMQP 836",
        "I m unable to access",
        "In the discussion around",
        "A spelling mistake was",
        "The AMQP 501 issue",
        "The issue AMQP 486"
    ]
    
    # Create binary labels for each defect type
    # In the image, most rows had type_bug and type_documentation set to 1, others to 0
    labels = np.zeros((len(reports), 7))
    labels[:, 2:4] = 1  # Set type_bug and type_documentation to 1
    
    # Column names from the image
    columns = ['type_blocker', 'type_regression', 'type_bug', 'type_documentation', 
               'type_enhancement', 'type_task', 'type_dependency_u']
    
    # Create DataFrame
    df = pd.DataFrame({
        'report': reports
    })
    
    for i, col in enumerate(columns):
        df[col] = labels[:, i]
    
    return df

# Load dataset
print("Loading dataset...")
# In a real implementation, you would load the CSV file:
# df = pd.read_csv('defect_prediction_dataset.csv')
df = create_simulated_dataset()

print("Dataset sample:")
print(df.head())

# Check label distribution
print("\nLabel distribution:")
label_cols = [col for col in df.columns if col.startswith('type_')]
label_dist = df[label_cols].sum().sort_values(ascending=False)
print(label_dist)

# Feature extraction using TF-IDF
print("\nExtracting features...")
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['report']).toarray()
y = df[label_cols].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Evaluation metrics for multi-label classification
def evaluate_multilabel_model(y_true, y_pred, k=3):
    hl = hamming_loss(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Precision@k (simplified version)
    # In a real implementation, you would compute this properly
    precision_at_k = precision_score(y_true, y_pred, average='micro')
    
    return {
        'Hamming Loss': hl,
        'Micro-F1': micro_f1,
        'Macro-F1': macro_f1,
        f'Precision@{k}': precision_at_k
    }

# 1. Logistic Regression (One-vs-Rest)
print("\nTraining Logistic Regression model...")
lr_model = MultiOutputClassifier(LogisticRegression(max_iter=1000))
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_metrics = evaluate_multilabel_model(y_test, lr_pred)

# 2. SVM (Multi-label version)
print("Training SVM model...")
svm_model = MultiOutputClassifier(SVC())
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_metrics = evaluate_multilabel_model(y_test, svm_pred)

# 3. Perceptron (standard)
print("Training Perceptron model...")
perceptron_model = MultiOutputClassifier(Perceptron(max_iter=1000))
perceptron_model.fit(X_train_scaled, y_train)
perceptron_pred = perceptron_model.predict(X_test_scaled)
perceptron_metrics = evaluate_multilabel_model(y_test, perceptron_pred)

# 4. Perceptron (online learning mode)
print("Training Perceptron in online learning mode...")
class OnlinePerceptron:
    def __init__(self, n_features, n_labels, learning_rate=0.01):
        self.weights = np.zeros((n_labels, n_features))
        self.biases = np.zeros(n_labels)
        self.learning_rate = learning_rate
        
    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.biases
        return (scores > 0).astype(int)
    
    def update(self, x, y):
        y_pred = self.predict(x.reshape(1, -1))[0]
        for i in range(len(y)):
            if y[i] != y_pred[i]:
                update = self.learning_rate * (y[i] - y_pred[i])
                self.weights[i] += update * x
                self.biases[i] += update
    
    def fit(self, X, y, epochs=1):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                self.update(X[i], y[i])

online_perceptron = OnlinePerceptron(X_train_scaled.shape[1], y_train.shape[1])
online_perceptron.fit(X_train_scaled, y_train, epochs=5)
online_perceptron_pred = online_perceptron.predict(X_test_scaled)
online_perceptron_metrics = evaluate_multilabel_model(y_test, online_perceptron_pred)

# 5. Deep Neural Network (DNN)
print("Training Deep Neural Network model...")
def create_dnn_model(input_dim, output_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

dnn_model = create_dnn_model(X_train_scaled.shape[1], y_train.shape[1])
dnn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=0)
dnn_prob = dnn_model.predict(X_test_scaled)
dnn_pred = (dnn_prob > 0.5).astype(int)
dnn_metrics = evaluate_multilabel_model(y_test, dnn_pred)

# Compare model performance
print("\nModel Performance Comparison:")
models = ['Logistic Regression', 'SVM', 'Perceptron', 'Online Perceptron', 'DNN']
metrics = ['Hamming Loss', 'Micro-F1', 'Macro-F1', 'Precision@3']
results = pd.DataFrame({
    'Logistic Regression': [lr_metrics[m] for m in metrics],
    'SVM': [svm_metrics[m] for m in metrics],
    'Perceptron': [perceptron_metrics[m] for m in metrics],
    'Online Perceptron': [online_perceptron_metrics[m] for m in metrics],
    'DNN': [dnn_metrics[m] for m in metrics]
}, index=metrics)

print(results)

# Visualize results
plt.figure(figsize=(12, 8))
results.plot(kind='bar')
plt.title('Multi-Label Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize label distribution in predictions
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# True labels distribution
axes[0].bar(label_cols, y_test.sum(axis=0))
axes[0].set_title('True Label Distribution')
axes[0].set_xticklabels(label_cols, rotation=90)

# Predicted labels distribution (using DNN as example)
axes[1].bar(label_cols, dnn_pred.sum(axis=0))
axes[1].set_title('Predicted Label Distribution (DNN)')
axes[1].set_xticklabels(label_cols, rotation=90)

plt.tight_layout()
plt.show()

print("\nMulti-Label Defect Prediction Pipeline Complete!")