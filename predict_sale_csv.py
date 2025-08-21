

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import logging
import os
from typing import Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ===============================
# CONFIGURATION
# ===============================
SEQUENCE_LENGTH = 7
CACHE_DIR = "cache"
AUTO_SAVE_CACHE = True  # Save predictions automatically

CSV_DIR = r"C:/Users/saash/OneDrive/Documents/updated_Data_csv"

MODEL_PATHS = {
    'hybrid_model': 'models/hybrid_sales_model_t.keras',
    'scaler_tab': 'models/scaler_tabular_t.pkl',
    'scaler_seq': 'models/scaler_sequence_t.pkl',
    'scaler_y': 'models/scaler_target_t.pkl'
}

# ===============================
# LOAD CSV DATA ONCE
# ===============================
print("🔄 Loading CSV files...")
ingredients_df = pd.read_csv(os.path.join(CSV_DIR, "ingredients.csv"))
ingredient_store_df = pd.read_csv(os.path.join(CSV_DIR, "ingredient_store.csv"))
orders_store_df = pd.read_csv(os.path.join(CSV_DIR, "orders_store.csv"))
order_store_detail_df = pd.read_csv(os.path.join(CSV_DIR, "order_store_detail.csv"))
pricelookup_df = pd.read_csv(os.path.join(CSV_DIR, "pricelookup.csv"))
pricelookup_ingredient_df = pd.read_csv(os.path.join(CSV_DIR, "pricelookup_ingredient.csv"))
pricelookup_store_df = pd.read_csv(os.path.join(CSV_DIR, "pricelookup_store.csv"))
unit_of_measures_df = pd.read_csv(os.path.join(CSV_DIR, "unit_of_measures.csv"))
print("✅ CSV files loaded successfully")

# ===============================
# LOAD ML ARTIFACTS
# ===============================
def load_artifacts():
    """Load ML model and scalers with error handling."""
    try:
        hybrid_model = load_model(MODEL_PATHS['hybrid_model'])
        scaler_tab = joblib.load(MODEL_PATHS['scaler_tab'])
        scaler_seq = joblib.load(MODEL_PATHS['scaler_seq'])
        scaler_y = joblib.load(MODEL_PATHS['scaler_y'])
        print("✅ ML artifacts loaded successfully")
        return hybrid_model, scaler_tab, scaler_seq, scaler_y
    except Exception as e:
        print(f"❌ Error loading ML artifacts: {e}")
        raise

# ===============================
# OPTIMIZED CSV DATA FETCHING
# ===============================
def fetch_product_data_from_csv(product_id, store_id):
    """
    Optimized CSV data fetching with better error handling and performance.
    """
    try:
        # Use pre-loaded DataFrames for better performance
        print(f"🔍 Fetching data for product {product_id} at store {store_id}")
        
        # Filter order details by product_id first (most restrictive filter)
        filtered_details = order_store_detail_df[
            order_store_detail_df["pricelookup_id"] == product_id
        ].copy()
        
        if filtered_details.empty:
            print(f"⚠️ No order details found for product {product_id}")
            return pd.DataFrame()
        
        # Merge with orders to get store_id and date info
        merged_df = pd.merge(
            filtered_details,
            orders_store_df[['id', 'store_id', 'created_at', 'total_item']],
            left_on="order_id",
            right_on="id",
            suffixes=("_detail", "_order")
        )
        
        # Filter by store_id
        merged_df = merged_df[merged_df["store_id"] == store_id]
        
        if merged_df.empty:
            print(f"⚠️ No data found for product {product_id} at store {store_id}")
            return pd.DataFrame()
        
        # Merge with pricelookup_store for product details
        merged_df = pd.merge(
            merged_df,
            pricelookup_store_df[['id', 'name', 'standard_price']],
            left_on="pricelookup_id",
            right_on="id",
            suffixes=("", "_price")
        )
        
        # Convert dates efficiently
        merged_df["created_at"] = pd.to_datetime(merged_df["created_at"], errors="coerce")
        merged_df = merged_df.dropna(subset=["created_at"])
        
        # Optimized aggregation
        result = (
            merged_df.groupby(["pricelookup_id", merged_df["created_at"].dt.date])
            .agg(
                product_name=("name", "first"),
                total_quantity_sold=("pricelookup_qty", "sum"),
                average_selling_price=("pricelookup_item_price", "mean"),
                standard_price=("standard_price", "first"),
                total_sale_value=("finalOriginalPrice", "sum"),
                average_items_in_order=("total_item", "mean"),
                order_count=("order_id", "count")  # New feature
            )
            .reset_index()
            .rename(columns={"created_at": "sale_date"})
        )
        
        print(f"✅ Data fetched: {len(result)} records")
        return result
        
    except Exception as e:
        print(f"❌ Error in fetch_product_data_from_csv: {e}")
        return pd.DataFrame()

# ===============================
# ENHANCED DATA CLEANING & FEATURE ENGINEERING
# ===============================
def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced data cleaning and feature engineering for better prediction accuracy.
    """
    if df.empty:
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # --- 1. Enhanced Data Cleaning ---
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values('sale_date')
    
    # Remove outliers using IQR method for quantity
    Q1 = df['total_quantity_sold'].quantile(0.25)
    Q3 = df['total_quantity_sold'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap outliers instead of removing them
    df['total_quantity_sold'] = np.clip(df['total_quantity_sold'], lower_bound, upper_bound)
    
    # Smart filling for missing values
    df['total_quantity_sold'] = df['total_quantity_sold'].fillna(0)
    
    # Use rolling median for skewed variables to avoid bias
    for col in ['average_selling_price', 'standard_price', 'average_items_in_order']:
        if col in df.columns:
            rolling_median = df[col].rolling(window=7, min_periods=1).median()
            df[col] = df[col].fillna(rolling_median)
            df[col] = df[col].fillna(df[col].median())
    
    # --- 2. Advanced Calendar Features ---
    df['day'] = df['sale_date'].dt.day
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['day_of_week'] = df['sale_date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['sale_date'].dt.quarter
    df['season'] = (df['month'] % 12 // 3 + 1).astype(int)
    df['day_of_year'] = df['sale_date'].dt.dayofyear
    df['week_of_year'] = df['sale_date'].dt.isocalendar().week.astype(int)
    df['month_end'] = df['sale_date'].dt.is_month_end.astype(int)
    df['month_start'] = df['sale_date'].dt.is_month_start.astype(int)
    
    # --- 3. Cyclical Encoding (prevents artificial jumps) ---
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    
    # --- 4. Advanced Price Features ---
    if 'standard_price' in df.columns and 'average_selling_price' in df.columns:
        df['price_ratio'] = np.where(
            df['standard_price'] > 0.01,
            df['average_selling_price'] / df['standard_price'],
            1.0
        )
        df['price_discount'] = np.where(
            df['standard_price'] > 0.01,
            (df['standard_price'] - df['average_selling_price']) / df['standard_price'],
            0.0
        )
    
    # --- 5. Advanced Lag Features ---
    # Multiple lag periods for better trend capture
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df['total_quantity_sold'].shift(lag).fillna(0)
    
    # --- 6. Rolling Statistics (trend indicators) ---
    for window in [3, 7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['total_quantity_sold'].rolling(window).mean().fillna(0)
        df[f'rolling_std_{window}'] = df['total_quantity_sold'].rolling(window).std().fillna(0)
        df[f'rolling_min_{window}'] = df['total_quantity_sold'].rolling(window).min().fillna(0)
        df[f'rolling_max_{window}'] = df['total_quantity_sold'].rolling(window).max().fillna(0)
    
    # --- 7. Growth and Momentum Features ---
    df['qty_pct_change'] = df['total_quantity_sold'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    df['qty_momentum_7'] = df['total_quantity_sold'] - df['lag_7']
    df['qty_momentum_30'] = df['total_quantity_sold'] - df['lag_30']
    
    # --- 8. Volatility Features ---
    df['volatility_7'] = df['total_quantity_sold'].rolling(7).std().fillna(0)
    df['volatility_30'] = df['total_quantity_sold'].rolling(30).std().fillna(0)
    
    # --- 9. Seasonal Decomposition Features ---
    # Simple seasonal patterns
    df['is_month_start'] = df['day'].isin([1, 2, 3]).astype(int)
    df['is_month_end'] = df['day'].isin([28, 29, 30, 31]).astype(int)
    df['is_quarter_start'] = df['month'].isin([1, 4, 7, 10]).astype(int)
    df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)
    
    # --- 10. Interaction Features ---
    df['weekend_price_effect'] = df['is_weekend'] * df['price_ratio']
    df['seasonal_price_effect'] = df['season'] * df['price_ratio']
    
    # --- 11. Holiday placeholder ---
    df['is_holiday'] = 0
    
    # Remove any remaining NaN values
    df = df.fillna(0)
    
    print(f"✅ Enhanced preprocessing complete: {df.shape[1]} features generated")
    return df

# ===============================
# OPTIMIZED FUTURE FEATURE GENERATION
# ===============================
def generate_future_features(start_date, days, product_df, holiday_dates=None, custom_dates=None):
    """
    Generate comprehensive future features for prediction.
    """
    # Calculate robust statistics (resistant to outliers)
    def robust_median(series):
        return np.nanmedian(series[series > 0]) if (series > 0).any() else 0
    
    # Get robust statistics from historical data
    stats_data = {
        'price_ratio': robust_median(product_df['price_ratio']) if 'price_ratio' in product_df.columns else 1.0,
        'average_selling_price': robust_median(product_df['average_selling_price']) if 'average_selling_price' in product_df.columns else 0,
        'standard_price': robust_median(product_df['standard_price']) if 'standard_price' in product_df.columns else 0,
        'average_items_in_order': robust_median(product_df['average_items_in_order']) if 'average_items_in_order' in product_df.columns else 0
    }
    
    # Generate dates
    if custom_dates:
        dates = custom_dates
    else:
        dates = [start_date + timedelta(days=i + 1) for i in range(days)]
    
    features = []
    
    for current_date in dates:
        is_holiday = 1 if holiday_dates and current_date in holiday_dates else 0
        is_weekend = 1 if current_date.weekday() >= 5 else 0
        
        # Basic calendar features (must match clean_and_preprocess exactly)
        day = current_date.day
        month = current_date.month
        year = current_date.year
        day_of_week = current_date.weekday()
        quarter = (month - 1) // 3 + 1
        season = (month % 12) // 3 + 1
        day_of_year = current_date.timetuple().tm_yday
        week_of_year = current_date.isocalendar()[1]
        
        # Month start/end indicators (must match clean_and_preprocess)
        month_end = 1 if current_date.is_month_end else 0
        month_start = 1 if current_date.is_month_start else 0
        
        # Cyclical features (must match clean_and_preprocess)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_sin = np.sin(2 * np.pi * day / 31)
        day_cos = np.cos(2 * np.pi * day / 31)
        
        # Seasonal patterns (must match clean_and_preprocess)
        is_month_start = 1 if day in [1, 2, 3] else 0
        is_month_end = 1 if day in [28, 29, 30, 31] else 0
        is_quarter_start = 1 if month in [1, 4, 7, 10] else 0
        is_quarter_end = 1 if month in [3, 6, 9, 12] else 0
        
        # Interaction features (must match clean_and_preprocess)
        weekend_price_effect = is_weekend * stats_data['price_ratio']
        seasonal_price_effect = season * stats_data['price_ratio']
        
        # Build feature vector in EXACT order as clean_and_preprocess
        # This order is critical for the model to work correctly
        feature_vector = [
            day, month, year, day_of_week, is_weekend, quarter, season,
            stats_data['price_ratio'], day_of_year, stats_data['average_selling_price'],
            stats_data['standard_price'], stats_data['average_items_in_order'],
            is_holiday, week_of_year, month_end, month_start, dow_sin, dow_cos,
            month_sin, month_cos, day_sin, day_cos, is_month_start, is_month_end,
            is_quarter_start, is_quarter_end, weekend_price_effect, seasonal_price_effect
        ]
        
        features.append(feature_vector)
    
    # Define column names in EXACT order as clean_and_preprocess
    columns = [
        'day', 'month', 'year', 'day_of_week', 'is_weekend', 'quarter', 'season',
        'price_ratio', 'day_of_year', 'average_selling_price', 'standard_price',
        'average_items_in_order', 'is_holiday', 'week_of_year', 'month_end', 'month_start',
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
        'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'weekend_price_effect', 'seasonal_price_effect'
    ]
    
    return pd.DataFrame(features, columns=columns), dates

# ===============================
# OPTIMIZED SALES PREDICTION
# ===============================
def predict_sales(product_id: int,
                  store_id: int,
                  days_to_forecast: int = 120,
                  start_date: Optional[str] = None,
                  holiday_dates: Optional[list] = None,
                  model=None, scaler_tab=None, scaler_seq=None, scaler_y=None
                  ) -> Tuple[pd.DataFrame, float, float, str]:
    """
    Optimized sales prediction with enhanced accuracy and performance.
    """
    print(f"🚀 Starting prediction for product {product_id} at store {store_id}")
    
    # --- Load artifacts if not provided ---
    if None in [model, scaler_tab, scaler_seq, scaler_y]:
        print("📦 Loading ML artifacts...")
        model, scaler_tab, scaler_seq, scaler_y = load_artifacts()
    
    # --- Load cache with versioning ---
    cache_file = os.path.join(CACHE_DIR, f"pred_v1.4_{product_id}_{store_id}.csv")
    cached_df = pd.DataFrame()
    
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=['date'])
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.date
            print(f"✅ Loaded {len(cached_df)} cached predictions")
        except Exception as e:
            print(f"⚠️ Cache read failed: {e}")
    
    # --- Load and preprocess product data ---
    print("📊 Loading product data...")
    raw_df = fetch_product_data_from_csv(product_id, store_id)
    if raw_df.empty:
        raise ValueError(f"No data found for product {product_id}")
    
    print("🔧 Preprocessing data...")
    product_df = clean_and_preprocess(raw_df)
    product_name = product_df['product_name'].iloc[0]
    standard_price = get_standard_price_from_csv(product_id)
    
    print(f"📈 Historical data: {len(product_df)} records")
    
    # --- Determine forecast range ---
    if start_date:
        start_date = pd.Timestamp(start_date).date()
        end_date = start_date + timedelta(days=days_to_forecast - 1)
    else:
        if not cached_df.empty:
            start_date = cached_df['date'].max() + timedelta(days=1)
        else:
            start_date = max(product_df['sale_date'].max().date(), datetime.today().date())
        end_date = start_date + timedelta(days=days_to_forecast - 1)
    
    all_dates = pd.date_range(start=start_date, end=end_date).date
    cached_dates = set(cached_df['date']) if not cached_df.empty else set()
    missing_dates = sorted([d for d in all_dates if d not in cached_dates])
    
    print(f"📅 Forecasting from {start_date} to {end_date} ({len(missing_dates)} new dates)")
    
    # --- Build optimized initial sequence ---
    print("🔄 Building prediction sequence...")
    if not cached_df.empty:
        seq_end_date = start_date - timedelta(days=1)
        seq_start_date = seq_end_date - timedelta(days=SEQUENCE_LENGTH - 1)
        
        cached_seq = cached_df[
            (cached_df['date'] >= seq_start_date) & (cached_df['date'] <= seq_end_date)
        ]['predicted_quantity'].values
        
        if len(cached_seq) >= SEQUENCE_LENGTH:
            sequence = cached_seq[-SEQUENCE_LENGTH:]
        else:
            needed = SEQUENCE_LENGTH - len(cached_seq)
            hist_seq = product_df['total_quantity_sold'].values[-needed:]
            sequence = np.concatenate([hist_seq, cached_seq])
    else:
        sequence = product_df['total_quantity_sold'].values[-SEQUENCE_LENGTH:]
    
    # Ensure sequence is properly sized and typed
    sequence = np.array(sequence, dtype=np.float32)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence = sequence[-SEQUENCE_LENGTH:]
    elif len(sequence) < SEQUENCE_LENGTH:
        sequence = np.pad(sequence, (SEQUENCE_LENGTH - len(sequence), 0), mode='constant')
    
    print(f"✅ Sequence built: {len(sequence)} values, range: {sequence.min():.2f} - {sequence.max():.2f}")
    
    # --- Generate predictions with batch processing ---
    new_predictions = []
    if missing_dates:
        print("🔮 Generating predictions...")
        
        # Generate features for all missing dates at once
        future_features, future_dates = generate_future_features(
            min(missing_dates) - timedelta(days=1),
            len(missing_dates),
            product_df,
            holiday_dates
        )
        
        current_sequence = sequence.copy()
        
        # Process predictions in smaller batches for memory efficiency
        batch_size = 10
        for i in range(0, len(missing_dates), batch_size):
            batch_end = min(i + batch_size, len(missing_dates))
            batch_dates = missing_dates[i:batch_end]
            batch_features = future_features.iloc[i:batch_end]
            
            print(f"   Processing batch {i//batch_size + 1}/{(len(missing_dates) + batch_size - 1)//batch_size}")
            
            for day, forecast_date in enumerate(batch_dates):
                # Get features for current day
                tab_features = batch_features.iloc[day].values.reshape(1, -1)
                
                # Ensure sequence is correct length
                if len(current_sequence) > SEQUENCE_LENGTH:
                    current_sequence = current_sequence[-SEQUENCE_LENGTH:]
                elif len(current_sequence) < SEQUENCE_LENGTH:
                    current_sequence = np.pad(current_sequence, (SEQUENCE_LENGTH - len(current_sequence), 0), mode='constant')
                
                # Scale inputs
                tab_features_scaled = scaler_tab.transform(tab_features)
                seq_scaled = scaler_seq.transform(current_sequence.reshape(-1, 1))
                seq_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, 1)
                
                # Predict
                pred_scaled = model.predict([tab_features_scaled, seq_scaled], verbose=0)
                pred_quantity = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
                pred_quantity = max(0, round(pred_quantity, 4))
                pred_amount = round(pred_quantity * standard_price, 2)
                
                # Store prediction
                new_predictions.append({
                    'date': forecast_date,
                    'predicted_quantity': pred_quantity,
                    'predicted_amount': pred_amount,
                    'is_weekend': 1 if forecast_date.weekday() >= 5 else 0,
                    'is_holiday': 1 if holiday_dates and forecast_date in holiday_dates else 0,
                    'standard_price': standard_price,
                    'product_name': product_name
                })
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred_quantity)
    
    print(f"✅ Generated {len(new_predictions)} new predictions")
    
    # --- Combine with cache efficiently ---
    if not cached_df.empty:
        relevant_cached = cached_df[cached_df['date'].isin(all_dates)]
        combined_df = pd.concat([relevant_cached, pd.DataFrame(new_predictions)], ignore_index=True)
    else:
        combined_df = pd.DataFrame(new_predictions)
    
    # Clean and sort
    combined_df = combined_df.sort_values('date').drop_duplicates('date', keep='last')
    combined_df = combined_df[combined_df['date'].isin(all_dates)]
    
    # --- Update cache ---
    if new_predictions and AUTO_SAVE_CACHE:
        try:
            updated_cache = pd.concat([cached_df, pd.DataFrame(new_predictions)], ignore_index=True)
            updated_cache = updated_cache.sort_values('date').drop_duplicates('date', keep='last')
            updated_cache.to_csv(cache_file, index=False)
            print(f"💾 Cache updated: {len(updated_cache)} total predictions")
        except Exception as e:
            print(f"⚠️ Cache update failed: {e}")
    
    # --- Final formatting ---
    combined_df['date'] = combined_df['date'].apply(lambda x: x.strftime('%d/%m/%Y'))
    total_forecasted_quantity = round(combined_df['predicted_quantity'].sum(), 2)
    total_forecasted_amount = round(combined_df['predicted_amount'].sum(), 2)
    
    print(f"💰 Total forecast: {total_forecasted_quantity} units, ${total_forecasted_amount}")
    
    # --- Enhanced verification and accuracy metrics ---
    try:
        print("📊 Calculating accuracy metrics...")
        actual_df = product_df[['sale_date', 'total_quantity_sold']].copy()
        actual_df.rename(columns={'sale_date': 'date', 'total_quantity_sold': 'actual_quantity'}, inplace=True)
        actual_df['date'] = actual_df['date'].dt.strftime('%d/%m/%Y')
        
        verify_df = combined_df.merge(actual_df, on='date', how='left')
        verify_df = verify_df.dropna(subset=['actual_quantity'])
        
        if not verify_df.empty:
            # Calculate multiple error metrics
            verify_df['error'] = verify_df['predicted_quantity'] - verify_df['actual_quantity']
            verify_df['abs_error'] = verify_df['error'].abs()
            verify_df['squared_error'] = verify_df['error'] ** 2
            
            # MAPE (Mean Absolute Percentage Error)
            verify_df['ape'] = np.where(
                verify_df['actual_quantity'] != 0,
                (verify_df['abs_error'] / verify_df['actual_quantity']) * 100,
                np.nan
            )
            
            # sMAPE (Symmetric MAPE)
            verify_df['smape'] = (
                200 * verify_df['abs_error'] /
                (verify_df['predicted_quantity'].abs() + verify_df['actual_quantity'].abs())
            )
            
            # RMSE (Root Mean Square Error)
            rmse = np.sqrt(verify_df['squared_error'].mean())
            
            # Calculate metrics
            mae = verify_df['abs_error'].mean()
            mape = verify_df['ape'].mean(skipna=True)
            smape = verify_df['smape'].mean()
            
            # Log detailed metrics
            with open("verification_log.txt", "a") as f:
                f.write(f"[{datetime.now()}] Product {product_id}-{store_id} | "
                        f"MAE: {mae:.4f}, MAPE: {mape:.2f}%, sMAPE: {smape:.2f}%, RMSE: {rmse:.4f} | "
                        f"Records: {len(verify_df)}\n")
            
            print(f"✅ Accuracy Metrics:")
            print(f"   MAE: {mae:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   sMAPE: {smape:.2f}%")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   Validation Records: {len(verify_df)}")
            
    except Exception as e:
        print(f"⚠️ Verification skipped: {e}")
    
    print(f"🎯 Prediction complete for {product_name}")
    return combined_df, total_forecasted_quantity, total_forecasted_amount, product_name

# ===============================
# OPTIMIZED UTILITY FUNCTIONS
# ===============================
def get_standard_price_from_csv(product_id):
    """Get standard price with error handling."""
    try:
        match = pricelookup_store_df[pricelookup_store_df['id'] == product_id]
        if not match.empty:
            return float(match.iloc[0]['standard_price'])
        return None
    except Exception as e:
        print(f"⚠️ Error getting standard price: {e}")
        return None

def get_ingredients_for_product_from_csv(store_product_id):
    """Get ingredients with optimized data access."""
    try:
        pls_row = pricelookup_store_df.loc[pricelookup_store_df['id'] == store_product_id]
        if pls_row.empty:
            return pd.DataFrame()
        
        pricelookup_id = pls_row.iloc[0]['pricelookup_id']
        
        # Get ingredients linked to this pricelookup_id
        pli_df = pricelookup_ingredient_df.loc[
            pricelookup_ingredient_df['pricelookup_id'] == pricelookup_id
        ]
        
        if pli_df.empty:
            return pd.DataFrame()
        
        # Merge with ingredients.csv to get ingredient names
        merged = pli_df.merge(
            ingredients_df[['id', 'name']],
            left_on="ingredient_id", 
            right_on="id", 
            how="left"
        )
        
        # Rename consistently
        merged.rename(columns={'name': 'ingredient_name', 'qty': 'qty_per_product_unit'}, inplace=True)
        
        # Add product info from pricelookup.csv
        product_name = pricelookup_df.loc[
            pricelookup_df['id'] == pricelookup_id, 'name'
        ].values[0]
        
        merged["product_id"] = pricelookup_id
        merged["product_name"] = product_name
        
        return merged[[
            "product_id",
            "product_name",
            "ingredient_id",
            "ingredient_name",
            "qty_per_product_unit"
        ]]
        
    except Exception as e:
        print(f"⚠️ Error getting ingredients: {e}")
        return pd.DataFrame()

def get_store_stock_for_product_ingredients_from_csv(store_product_id, store_id):
    """Get store stock with optimized data processing."""
    try:
        # Step 1: Get product from pricelookup_store_df
        pls = pricelookup_store_df[pricelookup_store_df['id'] == store_product_id]
        if pls.empty:
            return pd.DataFrame()
        
        p_id = pls.iloc[0]['pricelookup_id']
        
        # Step 2: Get ingredients for this product
        pli = (
            pricelookup_ingredient_df[pricelookup_ingredient_df['pricelookup_id'] == p_id]
            .rename(columns={'qty': 'qty_per_product_unit'})
            .copy()
        )
        
        if pli.empty:
            return pd.DataFrame()
        
        # Step 3: Merge with ingredient names
        ingredients_subset = ingredients_df[['id', 'name']].rename(
            columns={'id': 'ingredient_id', 'name': 'ingredient_name'}
        )
        merged = pli.merge(ingredients_subset, on="ingredient_id", how="left")
        
        # Step 4: Get stock for this store
        stock = ingredient_store_df[ingredient_store_df['store_id'] == store_id].copy()
        if 'id' in stock.columns:
            stock.rename(columns={'id': 'stock_id'}, inplace=True)
        
        merged = merged.merge(stock, on="ingredient_id", how="left")
        
        # Step 5: Merge with UOM table
        uom_df = unit_of_measures_df[['id', 'conversion_factor']].rename(columns={'id': 'uom_id'})
        merged = merged.merge(
            uom_df, 
            left_on="master_conversation_uom_type_id", 
            right_on="uom_id", 
            how="left"
        )
        
        # Step 6: Compute stock values with error handling
        merged["total_stockqty"] = (
            merged["stockqty"].fillna(0) + 
            merged["lastweek_left_stockqty"].fillna(0)
        )
        merged["conversion_factor"] = merged["conversion_factor"].fillna(1)
        
        # Step 7: Add product details
        merged["product_id"] = p_id
        merged["product_name"] = pricelookup_df.loc[
            pricelookup_df['id'] == p_id, 'name'
        ].values[0]
        
        return merged[[
            "product_id", 
            "product_name", 
            "ingredient_id", 
            "ingredient_name", 
            "qty_per_product_unit", 
            "total_stockqty", 
            "conversion_factor"
        ]]
        
    except Exception as e:
        print(f"⚠️ Error getting store stock: {e}")
        return pd.DataFrame()

def check_stock_sufficiency_from_csv(product_id, store_id, total_forecasted_quantity):
    """Check stock sufficiency with enhanced calculations."""
    try:
        ingredients_df_local = get_ingredients_for_product_from_csv(product_id)
        stock_df = get_store_stock_for_product_ingredients_from_csv(product_id, store_id)
        
        if ingredients_df_local.empty or stock_df.empty:
            return []
        
        # Merge data efficiently
        merged_df = pd.merge(
            ingredients_df_local[['ingredient_id', 'ingredient_name', 'qty_per_product_unit']],
            stock_df[['ingredient_id', 'ingredient_name', 'qty_per_product_unit', 'total_stockqty', 'conversion_factor']],
            on=['ingredient_id', 'ingredient_name', 'qty_per_product_unit'],
            how='left'
        )
        
        # Calculate requirements and stock levels
        merged_df['required_qty'] = merged_df['qty_per_product_unit'] * total_forecasted_quantity
        merged_df['adjusted_stockqty'] = merged_df['total_stockqty'] / merged_df['conversion_factor']
        
        # Determine status with safety margin (10% buffer)
        safety_margin = 1.1
        merged_df['status'] = np.where(
            merged_df['adjusted_stockqty'] >= (merged_df['required_qty'] * safety_margin), 
            'Sufficient', 
            'Needs Refill'
        )
        
        # Add stock utilization percentage
        merged_df['utilization_pct'] = np.where(
            merged_df['adjusted_stockqty'] > 0,
            (merged_df['required_qty'] / merged_df['adjusted_stockqty']) * 100,
            0
        )
        
        return merged_df[[
            'ingredient_name', 
            'qty_per_product_unit', 
            'required_qty', 
            'adjusted_stockqty', 
            'status',
            'utilization_pct'
        ]].to_dict(orient='records')
        
    except Exception as e:
        print(f"⚠️ Error checking stock sufficiency: {e}")
        return []

# ===============================
# PERFORMANCE MONITORING
# ===============================
def log_performance_metrics(func_name, start_time, end_time, success=True, error_msg=None):
    """Log performance metrics for monitoring."""
    duration = (end_time - start_time).total_seconds()
    status = "SUCCESS" if success else "FAILED"
    
    with open("performance_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] {func_name} | {status} | Duration: {duration:.2f}s")
        if error_msg:
            f.write(f" | Error: {error_msg}")
        f.write("\n")
    
    if success:
        print(f"⏱️ {func_name} completed in {duration:.2f}s")
    else:
        print(f"❌ {func_name} failed after {duration:.2f}s: {error_msg}")

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    print("🚀 Sales Prediction System - Optimized Version 1.4")
    print("=" * 60)
    
    # Test the system
    try:
        # Example usage
        product_id = 1
        store_id = 1
        days = 30
        
        print(f"🧪 Testing prediction for product {product_id} at store {store_id}")
        start_time = datetime.now()
        
        result_df, total_qty, total_amt, product_name = predict_sales(
            product_id=product_id,
            store_id=store_id,
            days_to_forecast=days
        )
        
        end_time = datetime.now()
        log_performance_metrics("predict_sales", start_time, end_time, success=True)
        
        print(f"✅ Test completed successfully!")
        print(f"📊 Results: {len(result_df)} predictions")
        print(f"💰 Total: {total_qty} units, ${total_amt}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()