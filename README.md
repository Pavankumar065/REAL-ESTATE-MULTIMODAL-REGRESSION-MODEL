# real-estate-multimodal-regression

A multimodal regression system that predicts residential property prices by combining structured housing data with satellite imagery.

---

## Overview
Traditional real estate valuation models rely solely on tabular attributes such as square footage, number of bedrooms, and construction quality. While effective, these features fail to capture important environmental and neighborhood-level factors that influence property value.

This project extends classical pricing models by integrating **satellite imagery** to encode visual context such as green cover, road density, and surrounding infrastructure. The final system combines numerical and visual signals to produce more accurate and explainable price predictions.

---

## Satellite Imagery
Satellite images are programmatically collected using the **Mapbox Static Images API**, based on latitude and longitude coordinates provided in the dataset.

- One satellite image per unique property  
- Images capture surrounding neighborhood context  
- Images are aligned with tabular records using property IDs  

A pretrained **ResNet-18** model is used as a fixed convolutional feature extractor. The final global average pooling layer outputs a **512-dimensional embedding** per image, representing high-level visual characteristics of each location.

---

## Modeling Approach

### Tabular Models
The following models are trained using structured housing features:
- **Linear Regression** (baseline)
- **Random Forest Regressor**
- **XGBoost Regressor**

XGBoost achieves the best performance among tabular models due to its ability to capture non-linear feature interactions.

### Image-Only Model
A separate XGBoost model is trained using only the CNN image embeddings. This model evaluates the predictive power of visual context independently from tabular data.

### Multimodal Fusion
Multimodal fusion is performed using **late feature fusion**:
- Tabular features are standardized using `StandardScaler`
- Image embeddings are extracted using ResNet-18
- Both feature sets are concatenated into a single feature vector
- A final XGBoost regressor is trained on the combined representation

This approach allows the model to learn complementary relationships between structured attributes and visual neighborhood cues.

---

## Model Performance

| Model                          | RMSE               | R² Score           |
|--------------------------------|--------------------|--------------------|
| Linear Regression (Tabular)    | 193064.8642878277  | 0.7029687257419828 |
| Random Forest (Tabular)        | 130138.00323328434 | 0.8650403423468688 |
| XGBoost (Tabular)              | 120336.86264815117 | 0.8846033811569214 |       
| XGBoost (Image Only)           | 318995.31241696957 | 0.1433895230293274 |
| **XGBoost (Multimodal)**       | 115952.7230555626  | 0.8868181705474854 |

The multimodal model consistently matches or outperforms tabular-only models, confirming that satellite imagery provides meaningful additional signal for property valuation.

---

## Explainability
Model predictions are interpreted using **Grad-CAM**, which highlights spatial regions of satellite images that contribute most strongly to price predictions. This provides visual insight into how environmental factors such as proximity to greenery, road networks, and water bodies influence housing value.

---

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, PyTorch, Torchvision, OpenCV, Matplotlib, Seaborn, Mapbox Static Images API

---

## Author
Pavan Kumar
