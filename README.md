# Machine Learning Lifecycle with TensorFlow

This repository is part of a collaborative student project for our university course in **Machine Learning**. It provides a complete walkthrough of the **machine learning lifecycle** using [TensorFlow](https://www.tensorflow.org/), including everything from data preparation to model evaluation and saving.

## 📋 Project Overview

This project demonstrates a complete end-to-end machine learning pipeline for **image classification** using TensorFlow. The pipeline includes:

- **Data Loading**: Processing images with Label Studio JSON annotations
- **Data Pipeline**: Creating efficient TensorFlow datasets with `tf.data`
- **TFRecord Creation**: Converting data to TensorFlow's optimized format
- **Model Training**: Both basic CNN and transfer learning with ResNet50
- **Model Evaluation**: Comprehensive performance analysis
- **Model Persistence**: Saving and loading trained models

## 📁 Project Structure (after running notebook)

```
├── ML_TF_notebook.ipynb          # Main Jupyter notebook with complete pipeline
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── data/                         # Dataset and processed files
│   ├── train_dataset.tfrecord    # Training data in TFRecord format
│   ├── eval_dataset.tfrecord     # Evaluation data in TFRecord format
│   ├── uni_test_train_ds/        # Training dataset
│   │   ├── images/               # Training images (PNG format)
│   │   └── labels/               # Label Studio JSON annotations
│   └── uni_test_eval_ds/         # Evaluation dataset
│       ├── images/               # Evaluation images (PNG format)
│       └── labels/               # Label Studio JSON annotations
└── saved_models/                 # Trained model files
    ├── basic_cnn.keras          # Basic CNN model
    └── resnet_model.keras       # ResNet50 transfer learning model
```

## 🎯 Contents

- **ML_TF_notebook.ipynb**  
  A comprehensive, step-by-step Jupyter Notebook covering:
  - Data loading from Label Studio JSON format
  - Image preprocessing and normalization
  - Creating efficient TFRecord datasets
  - Building a CNN from scratch
  - Transfer learning with pretrained ResNet50
  - Model training with train/validation splits
  - Comprehensive model evaluation and comparison
  - Model saving and loading best practices

## 🚀 Technologies

- **[TensorFlow](https://www.tensorflow.org/)** 2.x - Main ML framework
- **Python** ≥ 3.8 - Programming language
- **Jupyter Notebook** - Interactive development environment
- **Label Studio** format - For image annotations
- **NumPy** - Numerical computing

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd nak_machine_learning
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook ML_TF_notebook.ipynb
```

### 5. Run the Pipeline
Execute the cells in order to:
- Load and preprocess your image data
- Create TFRecord datasets
- Train both CNN and ResNet models
- Evaluate model performance
- Save trained models

## 📊 Dataset Format

This project expects:

- **Images**: PNG format in `data/*/images/` directories
- **Labels**: Label Studio JSON format in `data/*/labels/` directories
- **Structure**: Separate training and evaluation datasets

Example label format:
```json
[
  {
    "data": {"image": "/path/to/image.png"},
    "annotations": [{
      "result": [{
        "value": {"choices": ["class_name"]}
      }]
    }]
  }
]
```

## 🎓 Learning Objectives

This project demonstrates:

1. **Data Pipeline Design**: Efficient data loading and preprocessing
2. **TensorFlow Best Practices**: Using tf.data, TFRecords, and Keras
3. **Model Architecture**: Building CNNs and using transfer learning
4. **Training Strategies**: Train/validation splits and early stopping
5. **Model Evaluation**: Comprehensive performance analysis
6. **Production Readiness**: Model saving and loading

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Guide](https://keras.io/guides/)
- [tf.data Guide](https://www.tensorflow.org/guide/data)
- [TFRecord Tutorial](https://www.tensorflow.org/tutorials/load_data/tfrecord)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)


## 📞 Contact

If you have questions about this project or need help with the implementation, feel free to contact us directly through the university course channels.

---

**Note**: This project is designed for educational purposes as part of our Machine Learning course. The code prioritizes clarity and understanding over performance optimization.
