
# Plant Disease Detection Using Fine-Tuned MobileNetV2

## Project Overview

A deep learning application that classifies plant leaf images into
38 plant disease and healthy-leaf categories using a fine-tuned
MobileNetV2 model.

## Technologies Used

- Python
- TensorFlow and Keras
- MobileNetV2
- NumPy
- Pillow
- Streamlit

## Features

- Upload JPG, JPEG, and PNG leaf images
- Predict the most likely plant disease category
- Display model confidence
- Show the top three predictions
- Run locally through a Streamlit web interface

## Project Structure

```text
PlantDisease_Streamlit/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── mobilenetv2_plant_disease.keras
├── dataset/
│   └── class_names.json
└── assets/
```
## ⚠️ Important: Download the AI Model Files

The two LLM model files are not included in this GitHub repository because of their large file sizes.

To run the complete application:

1. Open the provided Google Colab notebook (`.ipynb`).
2. Run the notebook to obtain the two model files.
3. Download both files to your computer:
   - `Qwen3-4B-Q4_K_M.gguf`
   - `plant_disease_adapter.gguf`
4. Open the downloaded project folder.
5. Place both files inside the `models/` directory.

Your folder structure should look like this:

```text
PlantDisease_Streamlit/
├── app.py
├── models/
│   ├── mobilenetv2_plant_disease.keras
│   ├── Qwen3-4B-Q4_K_M.gguf
│   └── plant_disease_adapter.gguf
├── knowledge/
├── utils/
└── dataset/
```

**Note:** Both GGUF files are required for the fine-tuned AI Plant Disease Assistant. The CNN prediction feature can work independently.

## Setup and Run

1. Create and activate a Python virtual environment.
2. Install dependencies from requirements.txt.
3. Confirm the trained model and class names files are present.
4. Run the Streamlit application:

   streamlit run app.py

## Important Note

The model recognizes the 38 categories in its training dataset.
Predictions are estimates and should not be treated as guaranteed
agricultural diagnoses.
