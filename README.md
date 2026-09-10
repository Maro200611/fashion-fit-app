---
title: Fashion Fit Predictor
emoji: 👗
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: "1.45.1"
app_file: App.py
pinned: false
---

# Fashion Fit Predictor

Upload a photo — detect garments, estimate body type, and predict fit per item.

## Setup notes

Model artifacts (`best_xgboost_fit_model.pkl`, `numeric_scaler.pkl`, `categorical_encoder.pkl`,
`tfidf_vectorizer.pkl`, `fit_label_encoder.pkl`, `category_image_features.csv`, and optionally
`training_fallbacks.json`) must be placed in a `model_artifacts/` folder next to `App.py`.
