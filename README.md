# Hand Gesture Controlled Puzzle Game Research

A machine learning project for real-time hand gesture recognition using MediaPipe and XGBoost, designed for controlling puzzle games through hand gestures.

## 🎯 Project Overview

This project implements a robust hand gesture recognition system that can classify different hand gestures in real-time. The system uses MediaPipe for hand landmark detection and XGBoost for gesture classification, achieving high accuracy with minimal latency.

## 🏆 Model Performance

**Why XGBoost?**
- **High Accuracy**: Achieves 98% accuracy on test data
- **Real-time Performance**: Prediction time of only ~22ms, making it suitable for real-time applications
- **Efficiency**: Lightweight model that can run smoothly on standard hardware
- **Robustness**: Excellent performance with hand landmark features

## 🚀 Features

- **Real-time Hand Gesture Recognition**: Fast and accurate gesture classification
- **MediaPipe Integration**: Robust hand landmark detection
- **MLflow Tracking**: Complete experiment tracking and model versioning
- **Custom Preprocessing**: Wrist-centered normalization with middle finger scaling
- **Easy Training Pipeline**: Automated training with hyperparameter optimization
- **Model Persistence**: Save and load trained models with metadata

## 📊 Data Format

The system expects CSV data with hand landmarks in the following format:

```csv
x1,y1,z1,x2,y2,z2,...,x21,y21,z21,label
262.67,257.30,-3.64e-07,257.42,247.11,0.004,...,call
83.35,346.06,-2.34e-07,81.93,328.56,-0.011,...,call
187.76,260.24,-2.41e-07,195.46,241.51,-0.0001,...,call
```

- **21 hand landmarks** with x, y, z coordinates (63 features total)
- **Labels** representing different gesture classes

## 🛠️ Installation

### Quick Setup

```bash
git clone https://github.com/A190nux/Hand-Gesture-Controlled-Puzzle-Game-Research.git
cd Hand-Gesture-Controlled-Puzzle-Game-Research
```

### Install Dependencies

```bash
# Install all required packages with latest versions
python setup.py --install-latest

# Or install from requirements.txt
pip install -r requirements.txt
```

### Verify Installation

```bash
python setup.py --check
```

## 📁 Project Structure

```
Hand-Gesture-Controlled-Puzzle-Game-Research/
├── train_model.py          # Main training script
├── setup.py               # Environment setup and package management
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── Models/               # Saved models and artifacts
│   ├── hand_landmarker.task  # MediaPipe model (download required)
│   ├── *.pkl            # Trained XGBoost models
│   └── *.json           # Model metadata and class mappings
├── mlflow_data/          # MLflow experiment tracking
├── mlflow_artifacts/     # MLflow model artifacts
└── data/                # Training datasets
```

## 🏋️ Training

### Basic Training

```bash
# Train with your CSV dataset
python train_model.py --csv hand_landmarks_data.csv
```

### Advanced Training Options

```bash
# Custom parameters
python train_model.py \
    --csv your_dataset.csv \
    --test-size 0.2 \
    --random-state 42 \
    --max-depth 7 \
    --learning-rate 0.2 \
    --n-estimators 200

# Use default parameters instead of optimized ones
python train_model.py --csv your_dataset.csv --use-default-params

# Run without MLflow tracking
python train_model.py --csv your_dataset.csv --no-mlflow
```

### Training Pipeline

The training process follows these steps:

1. **Data Loading**: Load hand landmarks from CSV
2. **Data Splitting**: Stratified train-test split
3. **Custom Scaling**: Wrist-centered normalization with middle finger scaling
4. **Label Encoding**: Convert gesture names to numerical labels
5. **Model Training**: XGBoost with optimized hyperparameters
6. **Evaluation**: Comprehensive metrics and confusion matrix
7. **Model Saving**: Persist model, encoder, and metadata

## 📊 Model Architecture

### Preprocessing
- **Wrist Centering**: Use wrist (landmark 1) as origin point
- **Scale Normalization**: Scale using middle finger tip (landmark 13) distance
- **Z-coordinate Preservation**: Keep depth information unchanged

### XGBoost Configuration
```python
# Optimized parameters for best performance
{
    'colsample_bytree': 1.0,
    'learning_rate': 0.2,
    'max_depth': 7,
    'n_estimators': 200,
    'subsample': 0.8,
    'tree_method': 'hist'
}
```

## 📈 Experiment Tracking

The project uses MLflow for comprehensive experiment tracking:

```bash
# Start MLflow UI to view experiments
mlflow ui --backend-store-uri ./mlflow_data
```

Visit `http://localhost:5000` to view:
- Model performance metrics
- Hyperparameter comparisons
- Confusion matrices
- Model artifacts

## 🔮 Model Usage

### Loading a Trained Model

```python
import pickle
import pandas as pd

# Load model and encoder
with open('Models/xgboost_model_20240101_120000.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Models/label_encoder_20240101_120000.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict gesture
def predict_gesture(landmarks):
    # landmarks should be a DataFrame with 63 columns (x1,y1,z1,...,x21,y21,z21)
    scaled_landmarks = custom_scaling(landmarks)  # Apply same preprocessing
    prediction_encoded = model.predict(scaled_landmarks)
    prediction_label = label_encoder.inverse_transform(prediction_encoded)
    return prediction_label[0]
```

## 🎮 Integration with Puzzle Games

The trained model can be integrated into puzzle games for gesture-based control:

1. **Real-time Landmark Detection**: Use MediaPipe to detect hand landmarks
2. **Preprocessing**: Apply the same wrist-centering and scaling
3. **Gesture Classification**: Use trained XGBoost model for prediction
4. **Game Control**: Map gestures to game actions

## 📋 Requirements

### Core Dependencies
- **Python 3.8+**
- **MediaPipe 0.10.21**: Hand landmark detection
- **XGBoost 3.0.2**: Main classification model
- **scikit-learn 1.6.1**: Data preprocessing and metrics
- **MLflow 2.22.0**: Experiment tracking
- **OpenCV 4.11.0**: Computer vision utilities
- **NumPy, Pandas**: Data manipulation
- **Matplotlib, Seaborn**: Visualization

### Hardware Requirements
- **CPU**: Any modern processor (optimized for CPU inference)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB for models and data
- **Camera**: For real-time gesture capture (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe Team**: For the excellent hand landmark detection model
- **XGBoost Developers**: For the high-performance gradient boosting framework
- **MLflow Community**: For the comprehensive ML lifecycle management platform

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact the project maintainer.

---

**Note**: Make sure to download the MediaPipe hand landmark model (`hand_landmarker.task`) and place it in the `Models/` directory before running the training script. Download from: [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)# Hand-Gesture-Controlled-Puzzle-Game-Research