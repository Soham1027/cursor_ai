import os
import logging
from functools import lru_cache
from typing import Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta, date


# ===============================
# CONFIGURATION
# ===============================
SEQUENCE_LENGTH = 7
CACHE_DIR = "cache"
AUTO_SAVE_CACHE = True  # Save predictions automatically

# Allow overriding CSV path through environment variable to be portable
CSV_DIR = os.environ.get(
    "CSV_DIR",
    r"C:/Users/saash/OneDrive/Documents/updated_Data_csv"
)

MODEL_PATHS = {
    'hybrid_model': 'models/hybrid_sales_model_t.keras',
    'scaler_tab': 'models/scaler_tabular_t.pkl',
    'scaler_seq': 'models/scaler_sequence_t.pkl',
    'scaler_y': 'models/scaler_target_t.pkl'
}


logger = logging.getLogger(__name__)
os.makedirs(CACHE_DIR, exist_ok=True)


# ===============================
# LOAD CSV DATA ONCE (reused across calls)
# ===============================
def _safe_read_csv(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, parse_dates=parse_dates)
    except Exception as exc:
        logger.error(f"Failed to read CSV {path}: {exc}")
        return pd.DataFrame()


ingredients_df = _safe_read_csv(os.path.join(CSV_DIR, "ingredients.csv"))
ingredient_store_df = _safe_read_csv(os.path.join(CSV_DIR, "ingredient_store.csv"))
orders_store_df = _safe_read_csv(os.path.join(CSV_DIR, "orders_store.csv"), parse_dates=["created_at"])  # created_at used
order_store_detail_df = _safe_read_csv(os.path.join(CSV_DIR, "order_store_detail.csv"))
pricelookup_df = _safe_read_csv(os.path.join(CSV_DIR, "pricelookup.csv"))
pricelookup_ingredient_df = _safe_read_csv(os.path.join(CSV_DIR, "pricelookup_ingredient.csv"))
pricelookup_store_df = _safe_read_csv(os.path.join(CSV_DIR, "pricelookup_store.csv"))
unit_of_measures_df = _safe_read_csv(os.path.join(CSV_DIR, "unit_of_measures.csv"))


# Ensure expected columns exist to avoid KeyErrors later; log if missing
def _require_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    if df.empty:
        logger.warning(f"{df_name} is empty.")
        return
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(f"{df_name} missing columns: {missing}")


_require_columns(order_store_detail_df, [
    'order_id', 'pricelookup_id', 'pricelookup_qty', 'pricelookup_item_price', 'finalOriginalPrice'
], 'order_store_detail')
_require_columns(orders_store_df, ['id', 'store_id', 'created_at', 'total_item'], 'orders_store')
_require_columns(pricelookup_store_df, ['id', 'pricelookup_id', 'standard_price'], 'pricelookup_store')
_require_columns(pricelookup_df, ['id', 'name'], 'pricelookup')
_require_columns(pricelookup_ingredient_df, ['pricelookup_id', 'ingredient_id', 'qty'], 'pricelookup_ingredient')
_require_columns(ingredients_df, ['id', 'name'], 'ingredients')
_require_columns(ingredient_store_df, ['store_id', 'ingredient_id', 'stockqty', 'lastweek_left_stockqty'], 'ingredient_store')
_require_columns(unit_of_measures_df, ['id', 'conversion_factor'], 'unit_of_measures')


# ===============================
# LOAD ML ARTIFACTS
# ===============================
def load_artifacts():
    hybrid_model = load_model(MODEL_PATHS['hybrid_model'])
    scaler_tab = joblib.load(MODEL_PATHS['scaler_tab'])
    scaler_seq = joblib.load(MODEL_PATHS['scaler_seq'])
    scaler_y = joblib.load(MODEL_PATHS['scaler_y'])
    return hybrid_model, scaler_tab, scaler_seq, scaler_y


# ===============================
# CSV REPLACEMENTS FOR SQL - OPTIMIZED
# ===============================
@lru_cache(maxsize=256)
def fetch_product_data_from_csv(product_id: int, store_id: int) -> pd.DataFrame:
    """
    Fetch product sales and pricing data from preloaded CSV DataFrames.

    Args:
        product_id (int): pricelookup_store.id for the product in the given store
        store_id (int): store id

    Returns:
        pd.DataFrame: Aggregated by date with required features.
    """
    if pricelookup_store_df.empty or order_store_detail_df.empty or orders_store_df.empty:
        return pd.DataFrame()

    # Resolve main pricelookup_id from store-specific product id
    pls_row = pricelookup_store_df.loc[pricelookup_store_df['id'] == product_id]
    if pls_row.empty:
        return pd.DataFrame()

    main_pricelookup_id = int(pls_row.iloc[0]['pricelookup_id'])

    # Filter order details for this product (main id)
    details = order_store_detail_df.loc[
        order_store_detail_df['pricelookup_id'] == main_pricelookup_id,
        ['order_id', 'pricelookup_id', 'pricelookup_qty', 'pricelookup_item_price', 'finalOriginalPrice']
    ]
    if details.empty:
        return pd.DataFrame()

    # Join with orders to filter store and get created_at, total_item
    orders_needed = orders_store_df[['id', 'store_id', 'created_at', 'total_item']].rename(columns={'id': 'order_id'})
    merged = details.merge(orders_needed, on='order_id', how='inner')

    # Filter by store
    merged = merged.loc[merged['store_id'] == store_id]
    if merged.empty:
        return pd.DataFrame()

    # Attach product/store info (name, standard_price)
    product_name = None
    try:
        product_name = pricelookup_df.loc[pricelookup_df['id'] == main_pricelookup_id, 'name'].values[0]
    except Exception:
        product_name = None

    # Ensure created_at is datetime
    if not np.issubdtype(merged['created_at'].dtype, np.datetime64):
        merged['created_at'] = pd.to_datetime(merged['created_at'], errors='coerce')
    merged = merged.dropna(subset=['created_at'])
    if merged.empty:
        return pd.DataFrame()

    merged['sale_date'] = merged['created_at'].dt.normalize()

    # Aggregate per day
    agg = (
        merged.groupby('sale_date', as_index=False)
        .agg(
            pricelookup_id=('pricelookup_id', 'first'),
            product_name=('pricelookup_id', lambda _: product_name if product_name is not None else ''),
            total_quantity_sold=('pricelookup_qty', 'sum'),
            average_selling_price=('pricelookup_item_price', 'mean'),
            standard_price=('pricelookup_id', lambda _: float(pls_row.iloc[0]['standard_price']) if 'standard_price' in pls_row.columns else np.nan),
            total_sale_value=('finalOriginalPrice', 'sum'),
            average_items_in_order=('total_item', 'mean')
        )
        .sort_values('sale_date')
    )

    return agg


def get_standard_price_from_csv(product_id: int) -> Optional[float]:
    match = pricelookup_store_df[pricelookup_store_df['id'] == product_id]
    if not match.empty and 'standard_price' in match.columns:
        try:
            return float(match.iloc[0]['standard_price'])
        except Exception:
            return None
    return None


@lru_cache(maxsize=256)
def get_ingredients_for_product_from_csv(store_product_id: int) -> pd.DataFrame:
    pls_row = pricelookup_store_df.loc[pricelookup_store_df['id'] == store_product_id]
    if pls_row.empty:
        return pd.DataFrame()

    pricelookup_id = pls_row.iloc[0]['pricelookup_id']

    pli_df = pricelookup_ingredient_df.loc[
        pricelookup_ingredient_df['pricelookup_id'] == pricelookup_id
    ].copy()
    if pli_df.empty:
        return pd.DataFrame()

    merged = pli_df.merge(
        ingredients_df[['id', 'name']],
        left_on="ingredient_id",
        right_on="id",
        how="left"
    )

    merged.rename(columns={'name': 'ingredient_name', 'qty': 'qty_per_product_unit'}, inplace=True)

    product_name_vals = pricelookup_df.loc[
        pricelookup_df['id'] == pricelookup_id, 'name'
    ].values
    product_name = product_name_vals[0] if len(product_name_vals) else ''

    merged["product_id"] = pricelookup_id
    merged["product_name"] = product_name

    return merged[[
        "product_id",
        "product_name",
        "ingredient_id",
        "ingredient_name",
        "qty_per_product_unit"
    ]]


def get_store_stock_for_product_ingredients_from_csv(store_product_id: int, store_id: int) -> pd.DataFrame:
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

    # Step 3: Merge with ingredient names (only required columns)
    ingredients_subset = ingredients_df[['id', 'name']].rename(columns={'id': 'ingredient_id', 'name': 'ingredient_name'})
    merged = pli.merge(ingredients_subset, on="ingredient_id", how="left")

    # Step 4: Get stock for this store
    stock = ingredient_store_df[ingredient_store_df['store_id'] == store_id].copy()
    if stock.empty:
        # If no stock data for store, still return structure
        merged['total_stockqty'] = 0.0
    else:
        if 'id' in stock.columns:
            stock.rename(columns={'id': 'stock_id'}, inplace=True)
        merged = merged.merge(stock, on="ingredient_id", how="left")
        merged["total_stockqty"] = merged["stockqty"].fillna(0) + merged["lastweek_left_stockqty"].fillna(0)

    # Step 5: Merge with UOM table — only needed columns
    uom_df = unit_of_measures_df[['id', 'conversion_factor']].rename(columns={'id': 'uom_id'})
    merged = merged.merge(uom_df, left_on="master_conversation_uom_type_id", right_on="uom_id", how="left")

    # Step 6: Compute defaults
    merged["conversion_factor"] = merged["conversion_factor"].fillna(1)

    # Step 7: Add product details
    merged["product_id"] = p_id
    product_name_vals = pricelookup_df.loc[
        pricelookup_df['id'] == p_id, 'name'
    ].values
    merged["product_name"] = product_name_vals[0] if len(product_name_vals) else ''

    return merged[[
        "product_id",
        "product_name",
        "ingredient_id",
        "ingredient_name",
        "qty_per_product_unit",
        "total_stockqty",
        "conversion_factor"
    ]]


# ===============================
# DATA CLEANING
# ===============================
def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich sales dataframe with additional time-based and statistical features.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df = df.sort_values('sale_date')

    # Fill NA values more intelligently
    df['total_quantity_sold'] = df['total_quantity_sold'].fillna(0)

    # Median fill for skewed variables
    for col in ['average_selling_price', 'standard_price', 'average_items_in_order']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Calendar features
    df['day'] = df['sale_date'].dt.day
    df['month'] = df['sale_date'].dt.month
    df['year'] = df['sale_date'].dt.year
    df['day_of_week'] = df['sale_date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['quarter'] = df['sale_date'].dt.quarter
    df['season'] = (df['month'] % 12 // 3 + 1).astype(int)
    df['day_of_year'] = df['sale_date'].dt.dayofyear
    df['week_of_year'] = df['sale_date'].dt.isocalendar().week.astype(int)

    # Cyclical encoding for time (not used by model if scaler_tab didn't include; harmless to keep in df)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Price features
    df['price_ratio'] = np.where(
        df['standard_price'].fillna(0) > 0.01,
        df['average_selling_price'].fillna(0) / df['standard_price'].fillna(1),
        1.0
    )

    # Lag features
    df['lag_1'] = df['total_quantity_sold'].shift(1).fillna(0)
    df['lag_7'] = df['total_quantity_sold'].shift(7).fillna(df['lag_1'])
    df['lag_30'] = df['total_quantity_sold'].shift(30).fillna(df['lag_7'])

    # Rolling stats
    df['rolling_mean_7'] = df['total_quantity_sold'].rolling(7).mean().fillna(df['lag_1'])
    df['rolling_std_7'] = df['total_quantity_sold'].rolling(7).std().fillna(0)
    df['rolling_mean_30'] = df['total_quantity_sold'].rolling(30).mean().fillna(df['lag_7'])

    # Momentum
    df['qty_pct_change'] = df['total_quantity_sold'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

    # Placeholder for holidays (model expects a column; values set later for future)
    df['is_holiday'] = 0

    return df


# ===============================
# FUTURE FEATURE GENERATION
# ===============================
def _recent_or_median(series: pd.Series, window: int = 7) -> float:
    if series.empty:
        return 0.0
    try:
        recent = series.tail(window).mean()
        if pd.isna(recent) or np.isinf(recent):
            return float(series.median())
        return float(recent)
    except Exception:
        return float(series.median())


def generate_future_features(start_date: date,
                             days: int,
                             product_df: pd.DataFrame,
                             holiday_dates: Optional[List[date]] = None,
                             custom_dates: Optional[List[date]] = None) -> Tuple[pd.DataFrame, List[date]]:
    # Use recent values as a stronger prior than global median
    medians = {
        'price_ratio': _recent_or_median(product_df['price_ratio']) if 'price_ratio' in product_df.columns else 1.0,
        'average_selling_price': _recent_or_median(product_df['average_selling_price']) if 'average_selling_price' in product_df.columns else 0.0,
        'standard_price': _recent_or_median(product_df['standard_price']) if 'standard_price' in product_df.columns else 0.0,
        'average_items_in_order': _recent_or_median(product_df['average_items_in_order']) if 'average_items_in_order' in product_df.columns else 0.0,
    }

    dates = custom_dates if custom_dates else [start_date + timedelta(days=i + 1) for i in range(days)]
    holiday_set: Set[date] = set(holiday_dates or [])

    rows = []
    for current_date in dates:
        is_holiday = 1 if current_date in holiday_set else 0
        rows.append([
            current_date.day,
            current_date.month,
            current_date.year,
            current_date.weekday(),
            1 if current_date.weekday() >= 5 else 0,
            (current_date.month - 1) // 3 + 1,
            (current_date.month % 12) // 3 + 1,
            medians['price_ratio'],
            current_date.timetuple().tm_yday,
            medians['average_selling_price'],
            medians['standard_price'],
            medians['average_items_in_order'],
            is_holiday
        ])

    columns = [
        'day', 'month', 'year', 'day_of_week', 'is_weekend', 'quarter',
        'season', 'price_ratio', 'day_of_year', 'average_selling_price',
        'standard_price', 'average_items_in_order', 'is_holiday'
    ]
    return pd.DataFrame(rows, columns=columns), dates


# ===============================
# SALES PREDICTION (optimized inference loop)
# ===============================
def predict_sales(product_id: int,
                  store_id: int,
                  days_to_forecast: int = 120,
                  start_date: Optional[str] = None,
                  holiday_dates: Optional[List[date]] = None,
                  model=None, scaler_tab=None, scaler_seq=None, scaler_y=None
                  ) -> Tuple[pd.DataFrame, float, float, str]:
    """
    Predict future sales for a given product & store using tabular + sequence model.
    - Uses preloaded CSVs, caches join results, minimizes transforms in loop.
    - Preserves the same response schema as before.
    """

    # Load artifacts if not provided
    if None in [model, scaler_tab, scaler_seq, scaler_y]:
        model, scaler_tab, scaler_seq, scaler_y = load_artifacts()

    # Load cache
    cache_file = os.path.join(CACHE_DIR, f"pred_v1.3_{product_id}_{store_id}.csv")
    cached_df = pd.DataFrame()
    if os.path.exists(cache_file):
        try:
            cached_df = pd.read_csv(cache_file, parse_dates=['date'])
            cached_df['date'] = pd.to_datetime(cached_df['date']).dt.date
        except Exception as e:
            logger.warning(f"Cache read failed for {cache_file}: {e}")

    # Load product data
    raw_df = fetch_product_data_from_csv(product_id, store_id)
    if raw_df.empty:
        raise ValueError(f"No data found for product {product_id}")

    product_df = clean_and_preprocess(raw_df)
    product_name = product_df['product_name'].iloc[0] if 'product_name' in product_df.columns else ''
    standard_price_val = get_standard_price_from_csv(product_id)
    standard_price = standard_price_val if standard_price_val is not None else float(product_df.get('standard_price', pd.Series([0])).iloc[-1])

    # Forecast range
    if start_date:
        start_date_dt = pd.Timestamp(start_date).date()
        end_date_dt = start_date_dt + timedelta(days=days_to_forecast - 1)
    else:
        if not cached_df.empty:
            start_date_dt = cached_df['date'].max() + timedelta(days=1)
        else:
            start_date_dt = max(product_df['sale_date'].max().date(), datetime.today().date())
        end_date_dt = start_date_dt + timedelta(days=days_to_forecast - 1)

    all_dates = pd.date_range(start=start_date_dt, end=end_date_dt).date
    cached_dates = set(cached_df['date']) if not cached_df.empty else set()
    missing_dates = sorted([d for d in all_dates if d not in cached_dates])

    # Build initial sequence
    if not cached_df.empty:
        seq_end_date = start_date_dt - timedelta(days=1)
        seq_start_date = seq_end_date - timedelta(days=SEQUENCE_LENGTH - 1)
        cached_seq = cached_df[(cached_df['date'] >= seq_start_date) &
                               (cached_df['date'] <= seq_end_date)]['predicted_quantity'].values
        if len(cached_seq) >= SEQUENCE_LENGTH:
            sequence = cached_seq[-SEQUENCE_LENGTH:]
        else:
            needed = SEQUENCE_LENGTH - len(cached_seq)
            hist_seq = product_df['total_quantity_sold'].values[-needed:]
            sequence = np.concatenate([hist_seq, cached_seq])
    else:
        sequence = product_df['total_quantity_sold'].values[-SEQUENCE_LENGTH:]

    sequence = np.array(sequence, dtype=np.float32)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence = sequence[-SEQUENCE_LENGTH:]
    elif len(sequence) < SEQUENCE_LENGTH:
        sequence = np.pad(sequence, (SEQUENCE_LENGTH - len(sequence), 0), mode='constant')

    # Generate predictions
    new_predictions = []
    if missing_dates:
        future_features, future_dates = generate_future_features(
            min(missing_dates) - timedelta(days=1),
            len(missing_dates),
            product_df,
            holiday_dates
        )

        # Vectorize tabular scaling once
        tab_scaled_all = scaler_tab.transform(future_features.values)
        holiday_set = set(holiday_dates or [])

        current_sequence = sequence.copy()
        for day_idx, forecast_date in enumerate(missing_dates):
            tab_features_scaled = tab_scaled_all[[day_idx], :]

            # Scale sequence per step
            seq_scaled = scaler_seq.transform(current_sequence.reshape(-1, 1))
            seq_scaled = seq_scaled.reshape(1, SEQUENCE_LENGTH, 1)

            # Predict
            pred_scaled = model.predict([tab_features_scaled, seq_scaled], verbose=0)
            pred_quantity = float(scaler_y.inverse_transform(pred_scaled).flatten()[0])
            pred_quantity = max(0, round(pred_quantity, 4))
            pred_amount = round(pred_quantity * standard_price, 2)

            new_predictions.append({
                'date': forecast_date,
                'predicted_quantity': pred_quantity,
                'predicted_amount': pred_amount,
                'is_weekend': 1 if forecast_date.weekday() >= 5 else 0,
                'is_holiday': 1 if forecast_date in holiday_set else 0,
                'standard_price': standard_price,
                'product_name': product_name
            })

            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred_quantity)

    # Combine with cache
    if not cached_df.empty:
        relevant_cached = cached_df[cached_df['date'].isin(all_dates)]
        combined_df = pd.concat([relevant_cached, pd.DataFrame(new_predictions)])
    else:
        combined_df = pd.DataFrame(new_predictions)

    if not combined_df.empty and 'date' in combined_df.columns:
        combined_df = combined_df.sort_values('date').drop_duplicates('date')
        combined_df = combined_df[combined_df['date'].isin(all_dates)]

    # Update cache
    if new_predictions and AUTO_SAVE_CACHE:
        updated_cache = pd.concat([cached_df, pd.DataFrame(new_predictions)])
        updated_cache = updated_cache.sort_values('date').drop_duplicates('date')
        updated_cache.to_csv(cache_file, index=False)

    # Final formatting (keep dd/mm/YYYY strings as before)
    if not combined_df.empty:
        combined_df['date'] = combined_df['date'].apply(lambda x: x.strftime('%d/%m/%Y') if isinstance(x, (datetime, date)) else x)
    total_forecasted_quantity = round(float(combined_df['predicted_quantity'].sum()), 2) if not combined_df.empty else 0.0
    total_forecasted_amount = round(float(combined_df['predicted_amount'].sum()), 2) if not combined_df.empty else 0.0

    # Verification (best-effort, safe)
    try:
        actual_df = product_df[['sale_date', 'total_quantity_sold']].copy()
        actual_df.rename(columns={'sale_date': 'date', 'total_quantity_sold': 'actual_quantity'}, inplace=True)
        actual_df['date'] = actual_df['date'].dt.strftime('%d/%m/%Y')

        verify_df = combined_df.merge(actual_df, on='date', how='left').dropna(subset=['actual_quantity'])
        if not verify_df.empty:
            verify_df['error'] = verify_df['predicted_quantity'] - verify_df['actual_quantity']
            verify_df['abs_error'] = verify_df['error'].abs()
            verify_df['ape'] = np.where(verify_df['actual_quantity'] != 0,
                                        (verify_df['abs_error'] / verify_df['actual_quantity']) * 100,
                                        np.nan)
            verify_df['smape'] = (200 * verify_df['abs_error'] /
                                  (verify_df['predicted_quantity'].abs() + verify_df['actual_quantity'].abs()))

            mae = float(verify_df['abs_error'].mean())
            mape = float(verify_df['ape'].mean(skipna=True))
            smape = float(verify_df['smape'].mean())

            with open("verification_log.txt", "a") as f:
                f.write(f"[{datetime.now()}] Product {product_id}-{store_id} | MAE: {mae:.2f}, MAPE: {mape:.2f}%, sMAPE: {smape:.2f}%\n")
    except Exception as e:
        logger.debug(f"Verification skipped: {e}")

    return combined_df, total_forecasted_quantity, total_forecasted_amount, product_name


# ===============================
# STOCK CHECK
# ===============================
def check_stock_sufficiency_from_csv(product_id: int, store_id: int, total_forecasted_quantity: float):
    ingredients_df_local = get_ingredients_for_product_from_csv(product_id)
    stock_df = get_store_stock_for_product_ingredients_from_csv(product_id, store_id)

    if ingredients_df_local.empty:
        return []

    merged_df = pd.merge(
        ingredients_df_local[['ingredient_id', 'ingredient_name', 'qty_per_product_unit']],
        stock_df[['ingredient_id', 'ingredient_name', 'qty_per_product_unit', 'total_stockqty', 'conversion_factor']],
        on=['ingredient_id', 'ingredient_name', 'qty_per_product_unit'],
        how='left'
    )

    merged_df['total_stockqty'] = merged_df['total_stockqty'].fillna(0)
    merged_df['conversion_factor'] = merged_df['conversion_factor'].fillna(1).replace(0, 1)

    merged_df['required_qty'] = merged_df['qty_per_product_unit'] * float(total_forecasted_quantity)
    merged_df['adjusted_stockqty'] = merged_df['total_stockqty'] / merged_df['conversion_factor']
    merged_df['status'] = np.where(merged_df['adjusted_stockqty'] >= merged_df['required_qty'], 'Sufficient', 'Needs Refill')

    return merged_df[['ingredient_name', 'qty_per_product_unit', 'required_qty', 'adjusted_stockqty', 'status']].to_dict(orient='records')

