import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import Geopy tools for Reverse Geocoding
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Global variables for models, processors, baselines, and geocoder
cb_model = None
xgb_model = None
lgb_model = None
categorical_encoder = None
geolocator = None

weather_baseline = {}
training_feature_order = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cb_model, xgb_model, lgb_model, categorical_encoder, weather_baseline, training_feature_order, geolocator
    print("\n[Initialization] Setting up Geocoder and Model Ensembles...")
    
    # Initialize OpenStreetMap Nominatim Geocoder (Requires a unique User-Agent string)
    geolocator = Nominatim(user_agent="transit_predictor_application_v1")
    
    # 1. Load Dataset
    df = pd.read_excel("mEx 1.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Week_of_Year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Day_of_Week_Num'] = df['Date'].dt.dayofweek
    df['Hour_Int'] = pd.to_datetime(df['Hour'], format='%H:%M:%S').dt.hour

    df['Route_ID'] = (
        df['Source_Lat'].astype(str) + "_" + df['Source_Lon'].astype(str) + "___" +
        df['Destination_Lat'].astype(str) + "_" + df['Destination_Lon'].astype(str)
    )

    categorical_features = ['Day', 'Wind_Direction', 'Route_ID']
    numerical_features = [
        'Hour_Int', 'Source_Lat', 'Source_Lon', 'Destination_Lat', 'Destination_Lon',
        'Temperature_C', 'Humidity_%', 'Rain_mm', 'Cloud_Cover_%', 'Pressure_hPa',
        'Wind_Speed_kmh', 'Month', 'Week_of_Year', 'Day_of_Week_Num'
    ]

    training_feature_order = categorical_features + numerical_features
    X = df[training_feature_order].copy()
    y = df['Transit_Time_Min']

    for col in categorical_features:
        X[col] = X[col].astype(str)

    # 2. Fit the Ordinal Encoder
    categorical_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[categorical_features] = categorical_encoder.fit_transform(X[categorical_features])

    # 3. Train-Test Splits (CHANGED: 90% Training, 10% Testing)
    X_train_cb, X_test_cb, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train_enc, X_test_enc, _, _ = train_test_split(X_encoded, y, test_size=0.1, random_state=42)
    
    cat_indices = [X.columns.get_loc(col) for col in categorical_features]

    # --- ENSEMBLE TRAINING ---
    cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, eval_metric='MAE', random_seed=42, verbose=0)
    cb_model.fit(X_train_cb, y_train, cat_features=cat_indices, eval_set=(X_test_cb, y_test), early_stopping_rounds=50)

    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
    xgb_model.fit(X_train_enc, y_train, eval_set=[(X_test_enc, y_test)], verbose=False)

    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
    lgb_model.fit(X_train_enc, y_train, eval_set=[(X_test_enc, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])

    # --- ENSEMBLE ACCURACY EVALUATION ---
    cb_test_preds = cb_model.predict(X_test_cb)
    cb_mae = mean_absolute_error(y_test, cb_test_preds)
    
    xgb_test_preds = xgb_model.predict(X_test_enc)
    xgb_mae = mean_absolute_error(y_test, xgb_test_preds)
    
    lgb_test_preds = lgb_model.predict(X_test_enc)
    lgb_mae = mean_absolute_error(y_test, lgb_test_preds)
    
    ensemble_test_preds = (cb_test_preds + xgb_test_preds + lgb_test_preds) / 3.0
    ensemble_mae = mean_absolute_error(y_test, ensemble_test_preds)
    ensemble_r2 = r2_score(y_test, ensemble_test_preds)
    
    print("\n================ ENSEMBLE ACCURACY REPORT ================")
    print(f"CatBoost Model MAE : {cb_mae:.2f} Minutes")
    print(f"XGBoost Model MAE  : {xgb_mae:.2f} Minutes")
    print(f"LightGBM Model MAE : {lgb_mae:.2f} Minutes")
    print("----------------------------------------------------------")
    print(f"FINAL BLENDED ENSEMBLE MAE: {ensemble_mae:.2f} Minutes")
    print(f"Model Variance Score (R²): {ensemble_r2 * 100:.1f}%")
    print(f"Data Distribution Split   : 90% Train ({len(X_train_cb)} rows) | 10% Test ({len(X_test_cb)} rows)")
    print("==========================================================\n")

    # --- GENERATING THE EXCEL OUTPUT SHEETS ---
    print("[Excel Generation] Saving training outputs to 'Model_Training_Outputs.xlsx'...")
    
    # Sheet 1: Summary Metrics
    summary_metrics_data = {
        "Metric / Model Feature": [
            "CatBoost MAE (Minutes)", 
            "XGBoost MAE (Minutes)", 
            "LightGBM MAE (Minutes)", 
            "Final Blended Ensemble MAE (Minutes)", 
            "Ensemble R² Score (%)",
            "Total Training Samples (90%)",
            "Total Testing Samples (10%)"
        ],
        "Value": [
            round(cb_mae, 4),
            round(xgb_mae, 4),
            round(lgb_mae, 4),
            round(ensemble_mae, 4),
            round(ensemble_r2 * 100, 2),
            len(X_train_cb),
            len(X_test_cb)
        ]
    }
    df_summary = pd.DataFrame(summary_metrics_data)
    
    # Sheet 2: Test Record Targets vs Model Predictions
    df_predictions = X_test_cb.copy()
    df_predictions["Actual_Transit_Time"] = y_test.values
    df_predictions["CatBoost_Prediction"] = cb_test_preds
    df_predictions["XGBoost_Prediction"] = xgb_test_preds
    df_predictions["LightGBM_Prediction"] = lgb_test_preds
    df_predictions["Ensemble_Blended_Prediction"] = ensemble_test_preds
    df_predictions["Ensemble_Absolute_Error"] = np.abs(df_predictions["Actual_Transit_Time"] - df_predictions["Ensemble_Blended_Prediction"])
    
    # Write both structured frames out into dedicated multi-tabs
    with pd.ExcelWriter("Model_Training_Outputs.xlsx", engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Performance Summary", index=False)
        df_predictions.to_excel(writer, sheet_name="Test Set Predictions", index=False)
        
    print("[Excel Generation] Write complete. File is ready.\n")

    # Save weather configurations defaults
    weather_baseline = {
        "Temperature_C": float(df['Temperature_C'].mean()),
        "Humidity_%": float(df['Humidity_%'].mean()),
        "Rain_mm": 0.0,
        "Cloud_Cover_%": float(df['Cloud_Cover_%'].mean()),
        "Pressure_hPa": float(df['Pressure_hPa'].mean()),
        "Wind_Speed_kmh": float(df['Wind_Speed_kmh'].mean()),
        "Wind_Direction": str(df['Wind_Direction'].mode()[0]),
    }
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper function to convert raw coordinates to your specific address formats
def get_address_from_gps(lat: float, lon: float):
    try:
        # Request location object with language fallback
        location = geolocator.reverse((lat, lon), timeout=10, language='en')
        if location:
            full_address = location.address
            address_dict = location.raw.get('address', {})
            
            # Extract distinct components dynamically for the second string layer
            suburb = address_dict.get('suburb', address_dict.get('neighbourhood', ''))
            city = address_dict.get('city', address_dict.get('town', address_dict.get('suburb', '')))
            state = address_dict.get('state', '')
            country = address_dict.get('country', '')
            
            short_summary = f"{suburb} {city} {state} {country}".strip().replace("  ", " ")
            return full_address, short_summary
    except GeocoderTimedOut:
        pass
    
    return "Unknown Address Network Location", "Unknown City Base Address"

# Input coordinates schema
class CoordinateTimelineQuery(BaseModel):
    source_lat: float
    source_lon: float
    destination_lat: float
    destination_lon: float
    date_str: str

class CoordinateSingleHourQuery(BaseModel):
    source_lat: float
    source_lon: float
    destination_lat: float
    destination_lon: float
    date_str: str
    selected_hour: int = Field(..., ge=0, le=23)

# 1. LIVE REVERSE-GEOCODED SINGLE HOUR ENDPOINT
@app.post("/api/predict-single-hour")
async def get_single_hour_transit(query: CoordinateSingleHourQuery):
    from_full, from_short = get_address_from_gps(query.source_lat, query.source_lon)
    to_full, to_short = get_address_from_gps(query.destination_lat, query.destination_lon)
    
    target_date = pd.to_datetime(query.date_str)
    categorical_features = ['Day', 'Wind_Direction', 'Route_ID']
    
    single_trip_data = [{
        "Source_Lat": query.source_lat,
        "Source_Lon": query.source_lon,
        "Destination_Lat": query.destination_lat,
        "Destination_Lon": query.destination_lon,
        "Day": target_date.strftime("%A"),
        "Wind_Direction": str(weather_baseline["Wind_Direction"]),
        "Temperature_C": weather_baseline["Temperature_C"],
        "Humidity_%": weather_baseline["Humidity_%"],
        "Rain_mm": weather_baseline["Rain_mm"],
        "Cloud_Cover_%": weather_baseline["Cloud_Cover_%"],
        "Pressure_hPa": weather_baseline["Pressure_hPa"],
        "Wind_Speed_kmh": weather_baseline["Wind_Speed_kmh"],
        "Month": int(target_date.month),
        "Week_of_Year": int(target_date.isocalendar().week),
        "Day_of_Week_Num": int(target_date.dayofweek),
        "Hour_Int": int(query.selected_hour)
    }]
    
    base_df = pd.DataFrame(single_trip_data)
    base_df['Route_ID'] = (base_df['Source_Lat'].astype(str) + "_" + base_df['Source_Lon'].astype(str) + "___" + base_df['Destination_Lat'].astype(str) + "_" + base_df['Destination_Lon'].astype(str))
    
    for col in categorical_features:
        base_df[col] = base_df[col].astype(str)
        
    base_df = base_df[training_feature_order]
    encoded_df = base_df.copy()
    encoded_df[categorical_features] = categorical_encoder.transform(base_df[categorical_features])
    
    pred_cb = cb_model.predict(base_df)[0]
    pred_xgb = xgb_model.predict(encoded_df)[0]
    pred_lgb = lgb_model.predict(encoded_df)[0]
    
    final_ensemble_pred = max(0.0, round(float((pred_cb + pred_xgb + pred_lgb) / 3.0), 2))
    
    return {
        "from_location": {
            "latitude": query.source_lat,
            "longitude": query.source_lon,
            "full_address": from_full,
            "short_summary": from_short
        },
        "to_location": {
            "latitude": query.destination_lat,
            "longitude": query.destination_lon,
            "full_address": to_full,
            "short_summary": to_short
        },
        "departure_time": f"{str(query.selected_hour).zfill(2)}:00",
        "predicted_transit_time": f"{final_ensemble_pred} Minutes"
    }

# 2. LIVE REVERSE-GEOCODED 24-HOUR PROFILE ENDPOINT
@app.post("/api/predict-timeline")
async def get_24_hour_timeline(query: CoordinateTimelineQuery):
    from_full, from_short = get_address_from_gps(query.source_lat, query.source_lon)
    to_full, to_short = get_address_from_gps(query.destination_lat, query.destination_lon)
    
    target_date = pd.to_datetime(query.date_str)
    categorical_features = ['Day', 'Wind_Direction', 'Route_ID']
    
    synthetic_rows = []
    for hour in range(24):
        synthetic_rows.append({
            "Source_Lat": query.source_lat,
            "Source_Lon": query.source_lon,
            "Destination_Lat": query.destination_lat,
            "Destination_Lon": query.destination_lon,
            "Day": target_date.strftime("%A"),
            "Wind_Direction": str(weather_baseline["Wind_Direction"]),
            "Temperature_C": weather_baseline["Temperature_C"],
            "Humidity_%": weather_baseline["Humidity_%"],
            "Rain_mm": weather_baseline["Rain_mm"],
            "Cloud_Cover_%": weather_baseline["Cloud_Cover_%"],
            "Pressure_hPa": weather_baseline["Pressure_hPa"],
            "Wind_Speed_kmh": weather_baseline["Wind_Speed_kmh"],
            "Month": int(target_date.month),
            "Week_of_Year": int(target_date.isocalendar().week),
            "Day_of_Week_Num": int(target_date.dayofweek),
            "Hour_Int": int(hour)
        })
        
    base_df = pd.DataFrame(synthetic_rows)
    base_df['Route_ID'] = (base_df['Source_Lat'].astype(str) + "_" + base_df['Source_Lon'].astype(str) + "___" + base_df['Destination_Lat'].astype(str) + "_" + base_df['Destination_Lon'].astype(str))
    
    for col in categorical_features:
        base_df[col] = base_df[col].astype(str)
        
    base_df = base_df[training_feature_order]
    encoded_df = base_df.copy()
    encoded_df[categorical_features] = categorical_encoder.transform(base_df[categorical_features])

    preds_cb = cb_model.predict(base_df)
    preds_xgb = xgb_model.predict(encoded_df)
    preds_lgb = lgb_model.predict(encoded_df)

    final_ensemble_predictions = (preds_cb + preds_xgb + preds_lgb) / 3.0

    chart_data = []
    for hour in range(24):
        chart_data.append({
            "hour": f"{str(hour).zfill(2)}:00",
            "transit_time": max(0.0, round(float(final_ensemble_predictions[hour]), 2))
        })
        
    return {
        "from_location": {
            "latitude": query.source_lat,
            "longitude": query.source_lon,
            "full_address": from_full,
            "short_summary": from_short
        },
        "to_location": {
            "latitude": query.destination_lat,
            "longitude": query.destination_lon,
            "full_address": to_full,
            "short_summary": to_short
        },
        "timeline": chart_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)