# ===============================
# SYSTEM CONFIGURATION
# ===============================

import os
from typing import Dict, Any

class Config:
    """Configuration class for the sales prediction system."""
    
    # ===============================
    # ML MODEL CONFIGURATION
    # ===============================
    SEQUENCE_LENGTH = 7
    BATCH_SIZE = 10  # For prediction processing
    MODEL_VERBOSE = False  # Reduce TensorFlow output
    
    # ===============================
    # CACHE CONFIGURATION
    # ===============================
    CACHE_DIR = "cache"
    CACHE_VERSION = "v1.4"
    AUTO_SAVE_CACHE = True
    CACHE_CLEANUP_DAYS = 30  # Remove old cache files
    
    # ===============================
    # DATA PROCESSING CONFIGURATION
    # ===============================
    CSV_DIR = r"C:/Users/saash/OneDrive/Documents/updated_Data_csv"
    CHUNK_SIZE = 10000  # For large CSV processing
    LOW_MEMORY = False  # Trade memory for speed
    
    # ===============================
    # FEATURE ENGINEERING CONFIGURATION
    # ===============================
    # Lag periods for time series features
    LAG_PERIODS = [1, 2, 3, 7, 14, 30]
    
    # Rolling window sizes
    ROLLING_WINDOWS = [3, 7, 14, 30]
    
    # Outlier detection
    OUTLIER_IQR_MULTIPLIER = 1.5
    
    # Safety margin for stock calculations
    STOCK_SAFETY_MARGIN = 1.1  # 10% buffer
    
    # ===============================
    # PERFORMANCE CONFIGURATION
    # ===============================
    MAX_WORKERS = 4  # For parallel processing
    MEMORY_LIMIT_GB = 8  # Memory limit for processing
    TIMEOUT_SECONDS = 300  # Request timeout
    
    # ===============================
    # LOGGING CONFIGURATION
    # ===============================
    LOG_LEVEL = "INFO"
    LOG_FILE = "prediction_debug.log"
    PERFORMANCE_LOG = "performance_log.txt"
    VERIFICATION_LOG = "verification_log.txt"
    
    # ===============================
    # API CONFIGURATION
    # ===============================
    HOST = "0.0.0.0"
    PORT = 5000
    DEBUG = False
    THREADED = True
    
    # ===============================
    # JWT CONFIGURATION
    # ===============================
    JWT_SECRET_KEY = "secret_key"
    JWT_ACCESS_TOKEN_EXPIRES_DAYS = 30
    
    # ===============================
    # VALIDATION CONFIGURATION
    # ===============================
    MAX_PRODUCTS_PER_REQUEST = 50
    MAX_DAYS_FORECAST = 365
    MIN_HISTORICAL_DAYS = 30
    
    # ===============================
    # MODEL PATHS
    # ===============================
    MODEL_PATHS = {
        'hybrid_model': 'models/hybrid_sales_model_t.keras',
        'scaler_tab': 'models/scaler_tabular_t.pkl',
        'scaler_seq': 'models/scaler_sequence_t.pkl',
        'scaler_y': 'models/scaler_target_t.pkl'
    }
    
    # ===============================
    # ENVIRONMENT-SPECIFIC OVERRIDES
    # ===============================
    @classmethod
    def get_config(cls, environment: str = "production") -> Dict[str, Any]:
        """Get configuration based on environment."""
        config = {
            key: getattr(cls, key) 
            for key in dir(cls) 
            if not key.startswith('_') and not callable(getattr(cls, key))
        }
        
        # Environment-specific overrides
        if environment == "development":
            config.update({
                'DEBUG': True,
                'LOG_LEVEL': "DEBUG",
                'MEMORY_LIMIT_GB': 4,
                'MAX_WORKERS': 2
            })
        elif environment == "testing":
            config.update({
                'DEBUG': True,
                'LOG_LEVEL': "DEBUG",
                'CACHE_DIR': "test_cache",
                'TIMEOUT_SECONDS': 60
            })
        
        return config
    
    # ===============================
    # VALIDATION METHODS
    # ===============================
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings."""
        try:
            # Check required directories
            required_dirs = [cls.CACHE_DIR, os.path.dirname(cls.CSV_DIR)]
            for directory in required_dirs:
                if not os.path.exists(directory):
                    print(f"⚠️ Warning: Directory {directory} does not exist")
            
            # Check model files
            for model_path in cls.MODEL_PATHS.values():
                if not os.path.exists(model_path):
                    print(f"⚠️ Warning: Model file {model_path} does not exist")
            
            # Validate numeric ranges
            if cls.SEQUENCE_LENGTH < 1:
                raise ValueError("SEQUENCE_LENGTH must be >= 1")
            
            if cls.BATCH_SIZE < 1:
                raise ValueError("BATCH_SIZE must be >= 1")
            
            if cls.STOCK_SAFETY_MARGIN <= 0:
                raise ValueError("STOCK_SAFETY_MARGIN must be > 0")
            
            print("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False

# ===============================
# ENVIRONMENT CONFIGURATION
# ===============================
def get_environment_config():
    """Get configuration based on environment variables."""
    environment = os.getenv('FLASK_ENV', 'production')
    return Config.get_config(environment)

# ===============================
# PERFORMANCE TUNING
# ===============================
def optimize_pandas():
    """Apply pandas performance optimizations."""
    import pandas as pd
    
    # Enable pandas performance warnings
    pd.options.mode.chained_assignment = 'warn'
    
    # Set display options for better debugging
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)

def optimize_numpy():
    """Apply numpy performance optimizations."""
    import numpy as np
    
    # Set numpy threading
    os.environ['OMP_NUM_THREADS'] = str(Config.MAX_WORKERS)
    os.environ['MKL_NUM_THREADS'] = str(Config.MAX_WORKERS)

def optimize_tensorflow():
    """Apply TensorFlow performance optimizations."""
    import tensorflow as tf
    
    # Memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠️ GPU memory growth setup failed: {e}")
    
    # Threading
    tf.config.threading.set_inter_op_parallelism_threads(Config.MAX_WORKERS)
    tf.config.threading.set_intra_op_parallelism_threads(Config.MAX_WORKERS)
    
    # Mixed precision (if available)
    try:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed precision enabled")
    except:
        print("⚠️ Mixed precision not available")

# ===============================
# INITIALIZATION
# ===============================
def initialize_system():
    """Initialize system with optimizations."""
    print("🚀 Initializing Sales Prediction System...")
    
    # Apply optimizations
    optimize_pandas()
    optimize_numpy()
    optimize_tensorflow()
    
    # Validate configuration
    if not Config.validate_config():
        print("⚠️ Configuration validation failed, using defaults")
    
    # Create required directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("✅ System initialization complete")

if __name__ == "__main__":
    initialize_system()