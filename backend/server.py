from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import io
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Global variables for models and data
models_cache = {}
data_cache = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class DataUploadResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    rows: int
    columns: int
    message: str

class ManualDataEntry(BaseModel):
    stock_level_units: float
    storage_duration_days: float
    product_category: str
    product_value_usd: float
    storage_type: str
    inventory_turnover: float
    insurance_rate_percent: float
    obsolescence_risk: str
    storage_rent_usd_per_month: float
    handling_cost_per_unit: float
    security_level: str
    seasonality: str
    supplier_reliability: str

class PredictionRequest(BaseModel):
    data_id: str
    model_type: Optional[str] = "linear_regression"

class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    predictions: List[float]
    model_type: str

class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    predictions: List[str]
    model_type: str
    probabilities: Optional[List[List[float]]] = None

class TrainResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    message: str
    metrics: Dict[str, Any]

class EDAResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data_id: str
    stats: Dict[str, Any]
    distributions: Dict[str, Any]
    correlations: Optional[List[List[float]]] = None

# Helper functions
def create_cost_category(cost):
    """Create cost categories for classification"""
    if cost < 15000:
        return "Low"
    elif cost < 30000:
        return "Medium"
    else:
        return "High"

def preprocess_data(df):
    """Preprocess the dataset"""
    # Drop rows where target is missing
    if 'holding_cost_usd' in df.columns:
        df = df.dropna(subset=['holding_cost_usd'])
    
    # Identify column types
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target from feature columns if present
    if 'holding_cost_usd' in num_cols:
        num_cols.remove('holding_cost_usd')
    if 'cost_class' in cat_cols:
        cat_cols.remove('cost_class')
    
    return df, num_cols, cat_cols

def get_sample_data():
    """Generate sample inventory data"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'stock_level_units': np.random.randint(1000, 20000, n_samples).astype(float),
        'storage_duration_days': np.random.randint(30, 365, n_samples).astype(float),
        'product_category': np.random.choice(['Electronics', 'Automotive', 'Pharmaceuticals', 'Textiles', 'Food'], n_samples),
        'product_value_usd': np.random.uniform(100, 5000, n_samples),
        'storage_type': np.random.choice(['Climate Controlled', 'Outdoor', 'Standard Warehouse'], n_samples),
        'inventory_turnover': np.random.uniform(2, 12, n_samples),
        'insurance_rate_percent': np.random.uniform(0.5, 3, n_samples),
        'obsolescence_risk': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'storage_rent_usd_per_month': np.random.uniform(2, 10, n_samples),
        'handling_cost_per_unit': np.random.uniform(0.1, 5, n_samples),
        'security_level': np.random.choice(['Basic', 'Standard', 'High'], n_samples),
        'seasonality': np.random.choice(['Seasonal', 'Non-Seasonal'], n_samples),
        'supplier_reliability': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate holding cost based on features
    df['holding_cost_usd'] = (
        df['stock_level_units'] * df['product_value_usd'] * 0.01 +
        df['storage_duration_days'] * df['storage_rent_usd_per_month'] * 10 +
        df['stock_level_units'] * df['handling_cost_per_unit'] +
        df['product_value_usd'] * df['insurance_rate_percent'] * 0.01 * df['storage_duration_days'] / 30
    )
    
    return df

# Routes
@api_router.get("/")
async def root():
    return {"message": "Inventory Holding Cost Prediction API"}

@api_router.post("/upload", response_model=DataUploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Try reading as CSV first
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except:
            # Try reading as Excel
            df = pd.read_excel(io.BytesIO(contents))
        
        # Store data in cache
        data_id = str(uuid.uuid4())
        data_cache[data_id] = df.to_dict('records')
        
        # Store in MongoDB
        doc = {
            "data_id": data_id,
            "filename": file.filename,
            "rows": len(df),
            "columns": len(df.columns),
            "data": df.to_dict('records'),
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        await db.datasets.insert_one(doc)
        
        logger.info(f"Uploaded dataset {data_id} with {len(df)} rows and {len(df.columns)} columns")
        
        return DataUploadResponse(
            id=data_id,
            rows=len(df),
            columns=len(df.columns),
            message="Dataset uploaded successfully"
        )
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/manual-entry")
async def manual_entry(entry: ManualDataEntry):
    try:
        # Convert to dataframe
        df = pd.DataFrame([entry.model_dump()])
        
        # Store data in cache
        data_id = str(uuid.uuid4())
        data_cache[data_id] = df.to_dict('records')
        
        # Store in MongoDB
        doc = {
            "data_id": data_id,
            "type": "manual_entry",
            "rows": 1,
            "columns": len(df.columns),
            "data": df.to_dict('records'),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.datasets.insert_one(doc)
        
        return {"id": data_id, "message": "Data entry saved successfully"}
    except Exception as e:
        logger.error(f"Error in manual entry: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@api_router.get("/sample-data")
async def get_sample():
    try:
        df = get_sample_data()
        
        # Store data in cache
        data_id = str(uuid.uuid4())
        data_cache[data_id] = df.to_dict('records')
        
        return {
            "id": data_id,
            "rows": len(df),
            "columns": len(df.columns),
            "data": df.to_dict('records')[:10],  # Return first 10 rows
            "message": "Sample dataset loaded successfully"
        }
    except Exception as e:
        logger.error(f"Error generating sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/eda/{data_id}")
async def get_eda(data_id: str):
    try:
        # Get data from cache or database
        if data_id in data_cache:
            df = pd.DataFrame(data_cache[data_id])
        else:
            doc = await db.datasets.find_one({"data_id": data_id}, {"_id": 0})
            if not doc:
                raise HTTPException(status_code=404, detail="Dataset not found")
            df = pd.DataFrame(doc['data'])
        
        # Preprocess
        df, num_cols, cat_cols = preprocess_data(df)
        
        # Calculate statistics
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numerical_features": len(num_cols),
            "categorical_features": len(cat_cols)
        }
        
        # Generate distributions
        distributions = {}
        
        # Holding cost distribution
        if 'holding_cost_usd' in df.columns:
            hist, bins = np.histogram(df['holding_cost_usd'], bins=30)
            distributions['holding_cost'] = {
                "bins": bins.tolist(),
                "values": hist.tolist()
            }
        
        # Scatter plot data
        scatter_plots = {}
        if 'holding_cost_usd' in df.columns:
            for col in ['stock_level_units', 'storage_duration_days', 'product_value_usd', 'inventory_turnover']:
                if col in df.columns:
                    scatter_plots[col] = {
                        "x": df[col].fillna(0).tolist()[:100],  # Limit to 100 points for performance
                        "y": df['holding_cost_usd'].tolist()[:100]
                    }
        
        # Box plot data for categorical features
        box_plots = {}
        if 'holding_cost_usd' in df.columns:
            for col in cat_cols[:4]:  # Limit to first 4 categorical columns
                if col in df.columns:
                    groups = df.groupby(col)['holding_cost_usd'].apply(list).to_dict()
                    box_plots[col] = groups
        
        # Correlation matrix (numerical features only)
        correlations = None
        if num_cols and 'holding_cost_usd' in df.columns:
            corr_cols = [c for c in num_cols if c in df.columns] + ['holding_cost_usd']
            corr_matrix = df[corr_cols].corr()
            correlations = {
                "labels": corr_matrix.columns.tolist(),
                "data": corr_matrix.values.tolist()
            }
        
        return {
            "data_id": data_id,
            "stats": stats,
            "distributions": distributions,
            "scatter_plots": scatter_plots,
            "box_plots": box_plots,
            "correlations": correlations
        }
    except Exception as e:
        logger.error(f"Error in EDA: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/train/{data_id}")
async def train_models(data_id: str):
    try:
        # Get data
        if data_id in data_cache:
            df = pd.DataFrame(data_cache[data_id])
        else:
            doc = await db.datasets.find_one({"data_id": data_id}, {"_id": 0})
            if not doc:
                raise HTTPException(status_code=404, detail="Dataset not found")
            df = pd.DataFrame(doc['data'])
        
        if 'holding_cost_usd' not in df.columns:
            raise HTTPException(status_code=400, detail="Target column 'holding_cost_usd' not found")
        
        # Preprocess
        df, num_cols, cat_cols = preprocess_data(df)
        
        # Create cost categories for classification
        df['cost_class'] = df['holding_cost_usd'].apply(create_cost_category)
        
        # Prepare features and target for regression
        X = df.drop(columns=['holding_cost_usd', 'cost_class'])
        y_reg = df['holding_cost_usd']
        y_class = df['cost_class']
        
        # Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols)
            ]
        )
        
        # Train-test split
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        # Train regression models
        regression_metrics = {}
        
        # Linear Regression
        lr_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ])
        lr_pipe.fit(X_train, y_train_reg)
        y_pred_lr = lr_pipe.predict(X_test)
        
        regression_metrics['Linear Regression'] = {
            'r2_score': float(r2_score(y_test_reg, y_pred_lr)),
            'mae': float(mean_absolute_error(y_test_reg, y_pred_lr)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_reg, y_pred_lr)))
        }
        
        # Ridge Regression
        ridge_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))
        ])
        ridge_pipe.fit(X_train, y_train_reg)
        y_pred_ridge = ridge_pipe.predict(X_test)
        
        regression_metrics['Ridge Regression'] = {
            'r2_score': float(r2_score(y_test_reg, y_pred_ridge)),
            'mae': float(mean_absolute_error(y_test_reg, y_pred_ridge)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge)))
        }
        
        # MLP Regressor
        mlp_pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42))
        ])
        mlp_pipe.fit(X_train, y_train_reg)
        y_pred_mlp = mlp_pipe.predict(X_test)
        
        regression_metrics['Neural Network (MLP)'] = {
            'r2_score': float(r2_score(y_test_reg, y_pred_mlp)),
            'mae': float(mean_absolute_error(y_test_reg, y_pred_mlp)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_reg, y_pred_mlp)))
        }
        
        # Store trained models
        if data_id not in models_cache:
            models_cache[data_id] = {}
        
        models_cache[data_id]['linear_regression'] = lr_pipe
        models_cache[data_id]['ridge_regression'] = ridge_pipe
        models_cache[data_id]['mlp_regression'] = mlp_pipe
        models_cache[data_id]['preprocessor'] = preprocessor
        models_cache[data_id]['feature_columns'] = X.columns.tolist()
        models_cache[data_id]['num_cols'] = num_cols
        models_cache[data_id]['cat_cols'] = cat_cols
        
        # Classification metrics (simplified)
        le = LabelEncoder()
        y_class_encoded = le.fit_transform(y_class)
        _, _, y_train_class, y_test_class = train_test_split(
            X, y_class_encoded, test_size=0.2, random_state=42
        )
        
        # Store label encoder
        models_cache[data_id]['label_encoder'] = le
        
        classification_metrics = {
            'classes': le.classes_.tolist(),
            'distribution': {
                cls: int(np.sum(y_class_encoded == i))
                for i, cls in enumerate(le.classes_)
            }
        }
        
        # Store metrics in database
        metrics_doc = {
            "data_id": data_id,
            "regression_metrics": regression_metrics,
            "classification_metrics": classification_metrics,
            "trained_at": datetime.now(timezone.utc).isoformat()
        }
        await db.model_metrics.insert_one(metrics_doc)
        
        logger.info(f"Trained models for dataset {data_id}")
        
        return {
            "message": "Models trained successfully",
            "metrics": {
                "regression": regression_metrics,
                "classification": classification_metrics
            }
        }
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict-regression/{data_id}")
async def predict_regression(data_id: str, request: PredictionRequest):
    try:
        # Check if models are trained
        if data_id not in models_cache or 'linear_regression' not in models_cache[data_id]:
            raise HTTPException(status_code=400, detail="Models not trained. Please train models first.")
        
        model_type = request.model_type or 'linear_regression'
        model_key = f"{model_type}"
        
        if model_key not in models_cache[data_id]:
            raise HTTPException(status_code=400, detail=f"Model {model_type} not found")
        
        # Get test data
        if data_id in data_cache:
            df = pd.DataFrame(data_cache[data_id])
        else:
            doc = await db.datasets.find_one({"data_id": data_id}, {"_id": 0})
            if not doc:
                raise HTTPException(status_code=404, detail="Dataset not found")
            df = pd.DataFrame(doc['data'])
        
        # Preprocess
        df, _, _ = preprocess_data(df)
        X = df.drop(columns=['holding_cost_usd', 'cost_class'], errors='ignore')
        
        # Make predictions
        model = models_cache[data_id][model_key]
        predictions = model.predict(X)
        
        return {
            "predictions": predictions.tolist()[:100],  # Limit to first 100
            "model_type": model_type,
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error in regression prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/predict-classification/{data_id}")
async def predict_classification(data_id: str):
    try:
        # Check if models are trained
        if data_id not in models_cache or 'label_encoder' not in models_cache[data_id]:
            raise HTTPException(status_code=400, detail="Models not trained. Please train models first.")
        
        # Get test data
        if data_id in data_cache:
            df = pd.DataFrame(data_cache[data_id])
        else:
            doc = await db.datasets.find_one({"data_id": data_id}, {"_id": 0})
            if not doc:
                raise HTTPException(status_code=404, detail="Dataset not found")
            df = pd.DataFrame(doc['data'])
        
        # Preprocess
        df, _, _ = preprocess_data(df)
        
        # Create classifications based on holding cost
        if 'holding_cost_usd' in df.columns:
            predictions = df['holding_cost_usd'].apply(create_cost_category).tolist()
        else:
            # Use regression model to predict first
            if 'linear_regression' in models_cache[data_id]:
                X = df.drop(columns=['holding_cost_usd', 'cost_class'], errors='ignore')
                model = models_cache[data_id]['linear_regression']
                predicted_costs = model.predict(X)
                predictions = [create_cost_category(cost) for cost in predicted_costs]
            else:
                raise HTTPException(status_code=400, detail="Cannot perform classification without trained models")
        
        # Calculate distribution
        unique, counts = np.unique(predictions, return_counts=True)
        distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        return {
            "predictions": predictions[:100],  # Limit to first 100
            "model_type": "rule_based",
            "distribution": distribution,
            "count": len(predictions)
        }
    except Exception as e:
        logger.error(f"Error in classification prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/metrics/{data_id}")
async def get_metrics(data_id: str):
    try:
        # Get metrics from database
        doc = await db.model_metrics.find_one({"data_id": data_id}, {"_id": 0})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Metrics not found. Please train models first.")
        
        return doc
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
