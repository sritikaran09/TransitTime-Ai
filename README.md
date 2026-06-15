# 🚀 Transit Time Prediction

A FastAPI-based backend service that predicts transit times between geographic coordinates using an ensemble of CatBoost, XGBoost, and LightGBM models.

## ✨ Features

- 🤖 Ensemble learning with CatBoost, XGBoost, and LightGBM
- 📊 90/10 train-test validation
- 📑 Automatic Excel report generation
- 📍 Reverse geocoding with OpenStreetMap (Geopy)
- 🔮 Single-trip and 24-hour prediction endpoints
- ⚡ Real-time transit time forecasting

## 🛠️ Tech Stack

- Python
- FastAPI
- Pandas & NumPy
- Scikit-Learn
- CatBoost
- XGBoost
- LightGBM
- Geopy
- OpenPyXL

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/your-username/transit-time-predictor.git
cd transit-time-predictor
```

### Install dependencies

```bash
pip install numpy pandas scikit-learn catboost xgboost lightgbm fastapi uvicorn pydantic geopy openpyxl
```

## 📋 Dataset

Place `mEx 1.xlsx` in the project root directory.

Required columns:

```text
Date
Hour
Source_Lat
Source_Lon
Destination_Lat
Destination_Lon
Transit_Time_Min
Temperature_C
Humidity_%
Rain_mm
Cloud_Cover_%
Pressure_hPa
Wind_Speed_kmh
Wind_Direction
Day
```

## ▶️ Run the Application

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

## 📈 Model Pipeline

1. Load historical data
2. Generate temporal features
3. Create route identifiers
4. Train CatBoost, XGBoost, and LightGBM
5. Blend predictions
6. Evaluate model performance
7. Export results to Excel
8. Serve predictions through FastAPI

## 📄 Output

The application automatically generates:

```text
Model_Training_Outputs.xlsx
```

Containing:

- MAE
- RMSE
- R² Score
- Actual vs Predicted results

## 🌍 Reverse Geocoding

Converts latitude and longitude coordinates into human-readable addresses using OpenStreetMap Nominatim.

## 🔌 API Features

### Single Prediction
Predict transit time for a specific trip and hour.

### Daily Timeline
Generate hourly transit predictions for an entire day.

## 📊 Ensemble Formula

```text
Final Prediction =
(CatBoost + XGBoost + LightGBM) / 3
```


## 👨‍💻 Author

Sritikaran Sahoo

- GitHub: https://github.com/sritikaran09
- LinkedIn: www.linkedin.com/in/sritikaran-sahoo-a822661b0

## 📜 License

MIT License
