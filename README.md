# 🧬 Eczema Disease Classification with Gemini Vision API

This project uses **transfer learning** along with **Google's Gemini Vision API** to classify eczema from skin images. It combines the power of a pretrained CNN model and the Gemini multimodal API for improved classification and validation.

---

## 🔍 Overview

- 🔬 Classifies skin conditions with a focus on **eczema**.
- 🧠 Uses **transfer learning** to fine-tune a pretrained model.
- 🌐 Integrates **Gemini Vision API** for advanced image analysis or model-assisted validation.
- 📊 Includes utilities for data splitting, accuracy evaluation, and prediction.

---

## 📁 Project Structure

```
eczema-disease-classification/
│
├── train_model.py                 # Train the CNN model with transfer learning
├── main.py                        # Run predictions using model and Gemini Vision API
├── accuracy.py                    # Evaluate model accuracy
├── split_data.py                  # Split images into training and testing datasets
├── validatin_datasets.py          # Prepare/validate dataset structure
├── eczema_model_transfer_learning.h5  # Pre-trained model weights
├── requirement.txt                # Required Python packages
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/simonisavani/eczema-disease-classification.git
cd eczema-disease-classification
```

### 2. Install Dependencies
```bash
pip install -r requirement.txt
```

Make sure to include packages like `google-generativeai` or similar depending on the Gemini Vision API SDK used.

### 3. Set Up Gemini API

- Sign up and get your API key from [Google AI Studio](https://makersuite.google.com/)
- Set the API key as an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## 🧠 Model Training

Organize your dataset like this:

```
data/
├── train/
│   ├── eczema/
│   └── normal/
├── test/
│   ├── eczema/
│   └── normal/
```

Then train your model:

```bash
python train_model.py
```

---

## 🤖 Run Predictions (with Gemini)

```bash
python main.py
```

This script:
- Loads the trained model
- Classifies input images
- Uses the **Gemini Vision API** to validate or enhance predictions with natural language/image understanding

---

## 📊 Evaluation

```bash
python accuracy.py
```

This computes:
- Accuracy
- Precision
- Recall
- F1 Score

---

## 🌐 Gemini Vision API Integration

This project uses Gemini to:
- Validate predictions by asking Gemini to describe or assess the condition in an image.
- Cross-reference model predictions with Gemini's image-to-text feedback.

Feel free to use and adapt!
