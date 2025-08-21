from flask import Flask, request, jsonify
from predict_sale_csv import predict_sales, check_stock_sufficiency_from_csv, fetch_product_data_from_csv, get_standard_price_from_csv
from datetime import datetime, timedelta
import pandas as pd
import os
import logging
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
import time
from functools import wraps
import json

# Initialize Flask app
app = Flask(__name__)
app.config["JWT_SECRET_KEY"] = "secret_key"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(days=30)
jwt = JWTManager(app)

# Configuration
VALID_CREDENTIALS = {
    "username": "admin",
    "password": "Limer!@#123"
}
CACHE_VERSION = "v1.4"  # Updated cache version

# Set up logging with rotation
logging.basicConfig(
    filename='prediction_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.request_times = {}
        self.error_counts = {}
    
    def log_request(self, endpoint, duration, success=True, error_msg=None):
        if endpoint not in self.request_times:
            self.request_times[endpoint] = []
        self.request_times[endpoint].append(duration)
        
        if not success:
            if endpoint not in self.error_counts:
                self.error_counts[endpoint] = 0
            self.error_counts[endpoint] += 1
    
    def get_stats(self):
        stats = {}
        for endpoint, times in self.request_times.items():
            if times:
                stats[endpoint] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_requests': len(times),
                    'error_count': self.error_counts.get(endpoint, 0)
                }
        return stats

performance_monitor = PerformanceMonitor()

# ====== 🔧 Enhanced Utility Functions ======
def normalize_date(d):
    """Convert various date formats to date objects consistently."""
    if isinstance(d, str):
        try:
            # Try multiple date formats
            for fmt in ['%Y-%m-%d', '%d/%m/%Y', '%Y/%m/%d', '%d-%m-%Y']:
                try:
                    return datetime.strptime(d, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {d}")
        except ValueError as e:
            logging.error(f"Date parsing error: {e}")
            raise
    elif isinstance(d, datetime):
        return d.date()
    return d

def get_cache_path(product_id, store_id):
    """Get the path for cached predictions with versioning."""
    os.makedirs("cache", exist_ok=True)
    return f"cache/pred_{CACHE_VERSION}_{product_id}_{store_id}.csv"

def validate_cache(df):
    """Enhanced cache validation with better error handling."""
    required_cols = ['date', 'predicted_quantity', 'predicted_amount', 'product_name']
    
    # Check required columns
    if not all(col in df.columns for col in required_cols):
        logging.warning("Cache missing required columns")
        return False
    
    # Check for null values in critical columns
    critical_cols = ['date', 'predicted_quantity', 'predicted_amount']
    if df[critical_cols].isnull().values.any():
        logging.warning("Cache contains null values in critical columns")
        return False
    
    # Validate date format
    try:
        df['date'] = df['date'].apply(normalize_date)
    except Exception as e:
        logging.error(f"Cache date validation failed: {e}")
        return False
    
    # Validate numeric columns
    try:
        df['predicted_quantity'] = pd.to_numeric(df['predicted_quantity'], errors='coerce')
        df['predicted_amount'] = pd.to_numeric(df['predicted_amount'], errors='coerce')
        if df[['predicted_quantity', 'predicted_amount']].isnull().values.any():
            logging.warning("Cache contains invalid numeric values")
            return False
    except Exception as e:
        logging.error(f"Cache numeric validation failed: {e}")
        return False
    
    return True

def load_cached_predictions(product_id, store_id):
    """Load predictions from cache with enhanced validation."""
    path = get_cache_path(product_id, store_id)
    
    if os.path.exists(path):
        try:
            # Read CSV with optimized parsing
            df = pd.read_csv(
                path, 
                parse_dates=['date'],
                date_parser=lambda x: pd.to_datetime(x, dayfirst=True, errors='coerce'),
                low_memory=False
            )
            
            # Validate cache
            if not validate_cache(df):
                logging.warning(f"Invalid cache format for product {product_id}, removing file")
                os.remove(path)
                return pd.DataFrame()
            
            df['date'] = df['date'].apply(normalize_date)
            df = df.dropna(subset=['date'])
            
            logging.info(f"Successfully loaded {len(df)} cached predictions for product {product_id}")
            return df
            
        except Exception as e:
            logging.error(f"Cache load error for product {product_id}: {e}")
            # Remove corrupted cache file
            try:
                os.remove(path)
            except:
                pass
    
    return pd.DataFrame(columns=['date', 'predicted_quantity', 'predicted_amount', 'is_weekend', 'is_holiday', 'standard_price', 'product_name'])

def save_predictions_to_cache(product_id, store_id, df):
    """Save predictions to cache with enhanced error handling."""
    path = get_cache_path(product_id, store_id)
    
    try:
        # Ensure required columns exist
        required_cols = ['date', 'predicted_quantity', 'predicted_amount', 'product_name']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column for cache: {col}")
        
        # Convert dates to consistent format and sort
        df['date'] = df['date'].apply(normalize_date)
        df = df.sort_values('date')
        
        # Round numeric values for consistency
        numeric_cols = ['predicted_quantity', 'predicted_amount', 'standard_price']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].round(4)
        
        # Save to CSV with error handling
        df.to_csv(path, index=False)
        logging.info(f"Successfully saved cache for product {product_id}, store {store_id}")
        
    except Exception as e:
        logging.error(f"Failed to save cache for product {product_id}: {e}")
        raise

def performance_decorator(f):
    """Decorator to monitor endpoint performance."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        success = True
        error_msg = None
        
        try:
            result = f(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error_msg = str(e)
            logging.error(f"Error in {f.__name__}: {e}", exc_info=True)
            raise
        finally:
            duration = time.time() - start_time
            performance_monitor.log_request(f.__name__, duration, success, error_msg)
    
    return decorated_function

# ====== 🔐 Enhanced Login Endpoint ======
@app.route('/login', methods=['POST'])
@performance_decorator
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 0,
                'message': 'No data provided',
                'data': []
            }), 400
        
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({
                'status': 0,
                'message': 'Username and password are required',
                'data': []
            }), 400
        
        if username == VALID_CREDENTIALS['username'] and password == VALID_CREDENTIALS['password']:
            access_token = create_access_token(identity=username)
            logging.info(f"Successful login for user: {username}")
            return jsonify({
                'status': 1,
                'message': 'login data',
                'data': {'access_token': access_token}
            }), 200
        else:
            logging.warning(f"Failed login attempt for user: {username}")
            return jsonify({
                'status': 0,
                'message': 'Invalid credentials',
                'data': []
            }), 200
    except Exception as e:
        logging.error(f"Login error: {e}", exc_info=True)
        return jsonify({
            'status': 0,
            'message': 'Internal server error',
            'data': []
        }), 500

# ====== 📈 Optimized Prediction Endpoint ======
@app.route('/predict', methods=['POST'])
@jwt_required()
@performance_decorator
def predict():
    current_user = get_jwt_identity()
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 0,
                'message': 'No data provided',
                'data': []
            }), 400
        
        # Extract and validate input parameters
        product_ids = data.get('product_id', [])
        store_id = data.get('store_id')
        start_date_str = data.get('start_date')
        end_date_str = data.get('end_date')
        
        # Validate required parameters
        if not product_ids:
            return jsonify({
                'status': 0,
                'message': 'product_id is required',
                'data': []
            }), 400
        
        if not store_id:
            return jsonify({
                'status': 0,
                'message': 'store_id is required',
                'data': []
            }), 400
        
        # Ensure product_ids is a list
        if not isinstance(product_ids, list):
            product_ids = [product_ids]
        
        # Process holiday data with validation
        holidays_raw = data.get('filters', {}).get('holidays', [])
        holiday_dict = {}
        for h in holidays_raw:
            try:
                if isinstance(h, dict) and 'holiday_date' in h and 'holiday_name' in h:
                    date = normalize_date(h['holiday_date'])
                    holiday_dict[date] = h['holiday_name']
            except (KeyError, ValueError) as e:
                logging.warning(f"Invalid holiday data: {h}, error: {e}")
                continue
        
        holiday_dates = list(holiday_dict.keys())
        use_holiday_filter_only = len(holiday_dates) > 0
        
        # Date range processing with validation
        if use_holiday_filter_only:
            if not holiday_dates:
                return jsonify({
                    'status': 0,
                    'message': 'No valid holiday dates provided',
                    'data': []
                }), 400
            
            forecast_dates = sorted(holiday_dates)
            days_to_forecast = len(forecast_dates)
            start_date = forecast_dates[0]
            end_date = forecast_dates[-1]
            logging.info(f"Forecasting for {days_to_forecast} holiday days")
        else:
            if not (start_date_str and end_date_str):
                return jsonify({
                    'status': 2,
                    'message': 'Both start_date and end_date are required when holidays are not given.',
                    'data': []
                }), 400
            
            try:
                start_date = normalize_date(start_date_str)
                end_date = normalize_date(end_date_str)
            except ValueError as e:
                return jsonify({
                    'status': 0,
                    'message': f'Invalid date format: {e}',
                    'data': []
                }), 400
            
            if start_date > end_date:
                return jsonify({
                    'status': 2,
                    'message': 'start_date must be before end_date.',
                    'data': []
                }), 400
            
            days_to_forecast = (end_date - start_date).days + 1
            logging.info(f"Forecasting from {start_date} to {end_date} ({days_to_forecast} days)")
        
        # Process predictions for each product
        result = {}
        failed_products = []
        
        for product_id in product_ids:
            product_start_time = time.time()
            
            try:
                logging.info(f"Processing product {product_id} for store {store_id}")
                
                # Load and validate cache
                cached_df = load_cached_predictions(product_id, store_id)
                cached_dates = set(cached_df['date']) if not cached_df.empty else set()
                
                # Determine required dates
                if use_holiday_filter_only:
                    required_dates = set(holiday_dates)
                else:
                    required_dates = set(normalize_date(d) for d in pd.date_range(start=start_date, end=end_date))
                
                missing_dates = sorted(required_dates - cached_dates)
                logging.info(f"Product {product_id}: {len(missing_dates)} missing dates")
                
                # Prediction logic
                if not missing_dates:
                    # All dates available in cache
                    predictions_df = cached_df
                    total_forecasted_quantity = cached_df['predicted_quantity'].sum()
                    total_forecasted_amount = cached_df['predicted_amount'].sum()
                    product_name = cached_df['product_name'].iloc[0]
                    logging.info(f"Using cached predictions for product {product_id}")
                else:
                    # Generate new predictions
                    logging.info(f"Generating new predictions for product {product_id}")
                    predictions_df, total_forecasted_quantity, total_forecasted_amount, product_name = predict_sales(
                        product_id=product_id,
                        store_id=store_id,
                        days_to_forecast=days_to_forecast,
                        start_date=start_date,
                        holiday_dates=holiday_dates
                    )
                    
                    # Ensure product_name is included
                    predictions_df['product_name'] = product_name
                    
                    # Save to cache
                    try:
                        save_predictions_to_cache(product_id, store_id, predictions_df)
                    except Exception as e:
                        logging.error(f"Failed to save cache for product {product_id}: {e}")
                
                # Generate stock report
                stock_report = check_stock_sufficiency_from_csv(product_id, store_id, total_forecasted_quantity)
                
                # Round stock values for consistency
                for stock in stock_report:
                    for key in ['required_qty', 'adjusted_stockqty', 'qty_per_product_unit', 'utilization_pct']:
                        if key in stock:
                            stock[key] = round(stock[key], 2)
                
                # Filter predictions based on request
                if use_holiday_filter_only:
                    filtered_predictions = predictions_df[
                        predictions_df['date'].isin(holiday_dates)
                    ].copy()
                else:
                    filtered_predictions = predictions_df[
                        (predictions_df['date'] >= start_date) & 
                        (predictions_df['date'] <= end_date)
                    ].copy()
                
                # Add holiday names
                filtered_predictions['holiday_name'] = filtered_predictions['date'].map(holiday_dict)
                
                # Prepare response
                result[str(product_id)] = {
                    'predictions': filtered_predictions.to_dict(orient='records'),
                    'total_forecasted_quantity': round(float(total_forecasted_quantity), 2),
                    'total_forecasted_amount': round(float(total_forecasted_amount), 2),
                    'stock_sufficiency': stock_report,
                    'product_name': product_name
                }
                
                product_duration = time.time() - product_start_time
                logging.info(f"Product {product_id} processed in {product_duration:.2f}s")
                
            except Exception as e:
                product_duration = time.time() - product_start_time
                logging.error(f"Error processing product {product_id} after {product_duration:.2f}s: {str(e)}", exc_info=True)
                failed_products.append({
                    'product_id': product_id,
                    'error': str(e),
                    'processing_time': round(product_duration, 2)
                })
        
        # Calculate total processing time
        total_duration = time.time() - start_time
        
        # Prepare response
        response_data = {
            'status': 1,
            'message': 'predictions data' if result else 'No predictions available',
            'data': result,
            'failed_products': failed_products,
            'processing_time': round(total_duration, 2),
            'cache_version': CACHE_VERSION
        }
        
        logging.info(f"Prediction request completed in {total_duration:.2f}s for {len(product_ids)} products")
        return jsonify(response_data)
        
    except Exception as e:
        total_duration = time.time() - start_time
        logging.error(f"Unexpected error in predict endpoint after {total_duration:.2f}s: {str(e)}", exc_info=True)
        return jsonify({
            'status': 0,
            'message': 'Error occurred during prediction',
            'data': [str(e)],
            'processing_time': round(total_duration, 2)
        }), 500

# ====== 📊 Performance Monitoring Endpoint ======
@app.route('/performance', methods=['GET'])
@jwt_required()
def get_performance_stats():
    """Get performance statistics for monitoring."""
    try:
        stats = performance_monitor.get_stats()
        return jsonify({
            'status': 1,
            'message': 'Performance statistics',
            'data': stats
        })
    except Exception as e:
        logging.error(f"Error getting performance stats: {e}")
        return jsonify({
            'status': 0,
            'message': 'Error retrieving performance statistics',
            'data': []
        }), 500

# ====== 🧹 Cache Management Endpoint ======
@app.route('/cache/clear', methods=['POST'])
@jwt_required()
def clear_cache():
    """Clear all cached predictions."""
    try:
        cache_dir = "cache"
        if os.path.exists(cache_dir):
            cleared_files = 0
            for filename in os.listdir(cache_dir):
                if filename.startswith(f"pred_{CACHE_VERSION}_") and filename.endswith('.csv'):
                    file_path = os.path.join(cache_dir, filename)
                    os.remove(file_path)
                    cleared_files += 1
            
            logging.info(f"Cleared {cleared_files} cache files")
            return jsonify({
                'status': 1,
                'message': f'Cache cleared successfully. Removed {cleared_files} files.',
                'data': {'cleared_files': cleared_files}
            })
        else:
            return jsonify({
                'status': 1,
                'message': 'No cache directory found',
                'data': {'cleared_files': 0}
            })
    except Exception as e:
        logging.error(f"Error clearing cache: {e}")
        return jsonify({
            'status': 0,
            'message': 'Error clearing cache',
            'data': []
        }), 500

# ====== 🚀 Health Check Endpoint ======
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check if required directories exist
        cache_exists = os.path.exists("cache")
        models_exist = all(os.path.exists(path) for path in [
            'models/hybrid_sales_model_t.keras',
            'models/scaler_tabular_t.pkl',
            'models/scaler_sequence_t.pkl',
            'models/scaler_target_t.pkl'
        ])
        
        return jsonify({
            'status': 1,
            'message': 'System health check',
            'data': {
                'cache_directory': cache_exists,
                'model_files': models_exist,
                'timestamp': datetime.now().isoformat(),
                'version': CACHE_VERSION
            }
        })
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            'status': 0,
            'message': 'Health check failed',
            'data': []
        }), 500

# ====== 🚀 App Start ======
if __name__ == '__main__':
    print("🚀 Sales Prediction API - Optimized Version 1.4")
    print("=" * 60)
    print("📊 Performance monitoring enabled")
    print("💾 Enhanced caching system")
    print("🔧 Optimized data processing")
    print("=" * 60)
    
    # Create necessary directories
    os.makedirs("cache", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)