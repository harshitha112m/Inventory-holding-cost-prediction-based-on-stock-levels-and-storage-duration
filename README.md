# Inventory Holding Cost Prediction Dashboard

A full-stack machine learning application for predicting and categorizing inventory holding costs based on operational factors like stock levels, storage duration, product value, and more.

## Features

### 📊 Data Management
- **CSV/Excel Upload**: Drag-and-drop interface for uploading inventory data
- **Manual Data Entry**: Form-based entry for single records
- **Sample Dataset**: Pre-generated 100-sample dataset for testing

### 📈 Exploratory Data Analysis (EDA)
- Interactive visualizations using Chart.js
- Distribution histograms for holding costs
- Scatter plots showing relationships between features
- Categorical feature analysis with box plots
- Summary statistics dashboard

### 🤖 Machine Learning Models

#### Regression Models
- **Linear Regression**: Baseline model with interpretable coefficients
- **Ridge Regression**: L2 regularization to prevent overfitting
- **Neural Network (MLP)**: Multi-layer perceptron for non-linear relationships

#### Classification
- Cost categories: Low (<$15,000), Medium ($15,000-$30,000), High (>$30,000)
- Rule-based classification from regression predictions

### 📉 Model Evaluation
- R² Score comparison across models
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Interactive metric visualizations

## Tech Stack

### Frontend
- **React** 18.2.0 - UI framework
- **React Router** - Client-side routing
- **Chart.js** & **react-chartjs-2** - Data visualizations
- **React Dropzone** - File upload handling
- **Tailwind CSS** - Styling
- **Lucide React** - Icon library

### Backend
- **FastAPI** - Python web framework
- **scikit-learn** - Machine learning models
- **TensorFlow/Keras** - Neural network models
- **pandas** & **numpy** - Data processing
- **MongoDB** - Data storage
- **Motor** - Async MongoDB driver

### Design System
- Professional corporate style
- Navy (#0F172A) and Slate color palette
- Manrope font for headings, Inter for body text
- High-density dashboard layout with sidebar navigation

## Project Structure

```
/app/
├── backend/
│   ├── server.py              # FastAPI application with ML endpoints
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Backend environment variables
├── frontend/
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── components/
│   │   │   ├── Layout.js      # Main layout with sidebar
│   │   │   └── ui/            # Shadcn UI components
│   │   ├── pages/
│   │   │   ├── Dashboard.js   # Landing page
│   │   │   ├── DataUpload.js  # Data upload interface
│   │   │   ├── EDA.js         # Exploratory data analysis
│   │   │   ├── Predictions.js # Model predictions
│   │   │   └── Metrics.js     # Model evaluation metrics
│   │   ├── App.js             # Main app component
│   │   ├── App.css            # Component styles
│   │   └── index.css          # Global styles
│   ├── package.json           # Node dependencies
│   └── .env                   # Frontend environment variables
└── design_guidelines.json     # UI/UX design specifications
```

## API Endpoints

### Data Management
- `GET /api/` - API health check
- `POST /api/upload` - Upload CSV/Excel file
- `POST /api/manual-entry` - Submit manual data entry
- `GET /api/sample-data` - Load sample dataset

### Analysis & Training
- `GET /api/eda/{data_id}` - Get exploratory data analysis
- `POST /api/train/{data_id}` - Train all ML models

### Predictions
- `POST /api/predict-regression/{data_id}` - Get regression predictions
- `POST /api/predict-classification/{data_id}` - Get classification predictions
- `GET /api/metrics/{data_id}` - Get model evaluation metrics

## How to Run

### Prerequisites
- Python 3.11+
- Node.js 16+
- MongoDB

### Backend Setup
```bash
cd /app/backend

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup
```bash
cd /app/frontend

# Install dependencies
yarn install

# Start development server
yarn start
```

### Environment Variables

**Backend** (`/app/backend/.env`):
```
MONGO_URL=mongodb://localhost:27017
DB_NAME=test_database
CORS_ORIGINS=*
```

**Frontend** (`/app/frontend/.env`):
```
REACT_APP_BACKEND_URL=https://your-app-url.preview.emergentagent.com
WDS_SOCKET_PORT=443
ENABLE_HEALTH_CHECK=false
```

## Usage Guide

### 1. Upload Data
- Navigate to "Data Upload" from the sidebar
- Choose one of three options:
  - Upload CSV/Excel file via drag-and-drop
  - Enter data manually using the form
  - Load sample dataset (100 records)

### 2. Explore Data
- After upload, you're automatically redirected to the EDA page
- Review summary statistics (rows, columns, features)
- Analyze distribution charts and scatter plots
- Examine categorical feature impacts

### 3. Train Models
- Click "Train Models" button on the EDA page
- The system trains:
  - 3 regression models (Linear, Ridge, MLP)
  - Classification categories
- Training takes 10-20 seconds depending on dataset size

### 4. View Predictions
- Automatically redirected to Predictions page after training
- Select different regression models from dropdown
- View predicted holding costs with statistics
- See classification distribution (Low/Medium/High)

### 5. Evaluate Models
- Click "View Metrics" to see model performance
- Compare R² scores across models
- Review MAE and RMSE metrics
- Examine classification distribution

## ML Model Details

### Data Preprocessing
- Missing values: Mean imputation for numerical, mode for categorical
- Numerical features: StandardScaler normalization
- Categorical features: OneHotEncoder (for regression models)
- Train/test split: 80/20

### Model Parameters
- **Linear Regression**: Default scikit-learn parameters
- **Ridge Regression**: alpha=1.0
- **Neural Network**: 
  - Hidden layers: (64, 32)
  - Max iterations: 300
  - Activation: ReLU
  - Optimizer: Adam

### Performance Expectations
- R² scores typically range from 0.85 to 0.90 on sample data
- MAE around $70,000-75,000 on generated data
- Classification accuracy depends on cost distribution

## Example API Requests

### Upload CSV
```bash
curl -X POST "http://localhost:8001/api/upload" \\
  -F "file=@inventory_data.csv"
```

### Get Sample Data
```bash
curl -X GET "http://localhost:8001/api/sample-data"
```

### Train Models
```bash
curl -X POST "http://localhost:8001/api/train/{data_id}"
```

### Get Predictions
```bash
curl -X POST "http://localhost:8001/api/predict-regression/{data_id}" \\
  -H "Content-Type: application/json" \\
  -d '{"data_id": "your-data-id", "model_type": "linear_regression"}'
```

## Data Schema

### Input Features
- `stock_level_units` (float): Number of units in inventory
- `storage_duration_days` (float): Days in storage
- `product_category` (string): Electronics, Automotive, Pharmaceuticals, etc.
- `product_value_usd` (float): Unit value in USD
- `storage_type` (string): Climate Controlled, Outdoor, Standard Warehouse
- `inventory_turnover` (float): Turnover rate
- `insurance_rate_percent` (float): Insurance rate percentage
- `obsolescence_risk` (string): Low, Medium, High
- `storage_rent_usd_per_month` (float): Monthly storage cost
- `handling_cost_per_unit` (float): Per-unit handling cost
- `security_level` (string): Basic, Standard, High
- `seasonality` (string): Seasonal, Non-Seasonal
- `supplier_reliability` (string): Low, Medium, High

### Target Variable
- `holding_cost_usd` (float): Total holding cost in USD

## Design Guidelines

The application follows a professional corporate design system:
- **Colors**: Navy primary (#0F172A), Sky accent (#0EA5E9)
- **Typography**: Manrope for headings, Inter for body
- **Layout**: Fixed sidebar with scrollable main content
- **Components**: Sharp corners, subtle shadows, hover transitions
- **Charts**: Clean Chart.js visualizations with consistent styling

## Troubleshooting

### Models not training
- Ensure target column `holding_cost_usd` exists in your data
- Check that numerical columns are properly formatted
- Verify MongoDB connection is active

### Predictions showing errors
- Train models first before requesting predictions
- Ensure the data_id matches the uploaded dataset
- Check backend logs for detailed error messages

### Frontend not connecting to backend
- Verify REACT_APP_BACKEND_URL in frontend/.env
- Ensure backend is running on port 8001
- Check CORS settings in backend/.env

## Future Enhancements

- Real-time model retraining with new data
- Advanced feature engineering and selection
- Ensemble methods (Random Forest, Gradient Boosting)
- Time-series forecasting for seasonal trends
- Export predictions to CSV/Excel
- User authentication and multi-tenancy
- Model versioning and A/B testing

## License

This project is created for educational and demonstration purposes.

## Credits

Built with:
- FastAPI for high-performance Python backend
- React for modern UI development
- scikit-learn & TensorFlow for machine learning
- Chart.js for beautiful data visualizations
- MongoDB for flexible data storage

---

**Note**: This is a machine learning demonstration project. For production use, implement proper data validation, security measures, and error handling.
