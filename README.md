
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

## Setup and Run

1. Create and activate a Python virtual environment.
2. Install dependencies from requirements.txt.
3. Confirm the trained model and class names files are present.
4. Run the Streamlit application:

   streamlit run app.py

## ⚠️ Disclaimer — Please Read Before Using

**Plant Disease Detection Using Fine-Tuned MobileNetV2**

This application uses a deep learning model trained and evaluated on the **PlantVillage dataset**, containing images belonging to 38 plant and leaf-condition categories across 14 plant types.

**Supported plants:** Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

**Important limitations:**

- **Limited plant coverage:** The model is designed to classify images into the 38 categories it was trained on. It is not designed to identify every plant species or detect every plant disease.
- **Unsupported plants and diseases:** If you upload a leaf from a plant that is not represented in the training categories, or a disease that the model has not learned, the prediction may be incorrect or misleading.
- **Image quality matters:** Blurry, poorly lit, distant, damaged, or partially visible leaves may affect prediction accuracy.
- **Dataset versus real-world conditions:** The model was trained on PlantVillage images. Photos captured in real gardens, farms, or different lighting and background conditions may produce less reliable results.
- **Confidence is not certainty:** The displayed confidence score represents the model's predicted probability for a category. It does not guarantee that the prediction is correct. The model may assign a high score even to an unfamiliar plant or unsupported disease.
- **Not a substitute for expert advice:** This application is intended for educational and demonstration purposes. Do not rely solely on its predictions to make important crop-treatment or pesticide decisions. Consult a qualified agricultural expert when necessary.

**Recommended usage:** Upload a clear, well-lit image of a single leaf from one of the 14 supported plant types. Interpret the result only within the model's supported categories.

By using this application, users should understand that its predictions are estimates and that incorrect classifications are possible.


## Important Note

The model recognizes the 38 categories in its training dataset.
Predictions are estimates and should not be treated as guaranteed
agricultural diagnoses.
