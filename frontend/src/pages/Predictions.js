import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader2, TrendingUp, Activity, PieChart } from 'lucide-react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Predictions = () => {
  const { dataId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [regressionData, setRegressionData] = useState(null);
  const [classificationData, setClassificationData] = useState(null);
  const [error, setError] = useState('');
  const [modelType, setModelType] = useState('linear_regression');

  useEffect(() => {
    fetchPredictions();
  }, [dataId, modelType]);

  const fetchPredictions = async () => {
    try {
      setLoading(true);
      
      // Fetch regression predictions
      const regResponse = await axios.post(`${API}/predict-regression/${dataId}`, {
        data_id: dataId,
        model_type: modelType
      });
      setRegressionData(regResponse.data);

      // Fetch classification predictions
      const classResponse = await axios.post(`${API}/predict-classification/${dataId}`);
      setClassificationData(classResponse.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const viewMetrics = () => {
    navigate(`/metrics/${dataId}`);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-slate-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 bg-red-50 border-red-200">
        <p className="text-red-800">Error: {error}</p>
        <p className="text-sm text-red-600 mt-2">Make sure models are trained first.</p>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          boxWidth: 8,
          font: { family: 'Inter', size: 12 }
        }
      },
      tooltip: {
        backgroundColor: '#0F172A',
        titleFont: { family: 'Manrope', size: 13 },
        bodyFont: { family: 'Inter', size: 12 },
        padding: 12,
        cornerRadius: 4,
        displayColors: false
      }
    },
    scales: {
      x: {
        grid: { display: false },
        ticks: { font: { family: 'Inter', size: 11 }, color: '#64748B' }
      },
      y: {
        grid: { color: '#E2E8F0', borderDash: [4, 4] },
        ticks: { font: { family: 'Inter', size: 11 }, color: '#64748B' },
        border: { display: false }
      }
    }
  };

  const regressionChart = regressionData ? {
    labels: regressionData.predictions.slice(0, 20).map((_, i) => `Sample ${i + 1}`),
    datasets: [{
      label: 'Predicted Holding Cost (USD)',
      data: regressionData.predictions.slice(0, 20),
      backgroundColor: '#0EA5E9',
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  const classificationChart = classificationData?.distribution ? {
    labels: Object.keys(classificationData.distribution),
    datasets: [{
      label: 'Count',
      data: Object.values(classificationData.distribution),
      backgroundColor: ['#10B981', '#0EA5E9', '#F59E0B'],
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  const getCategoryColor = (category) => {
    switch (category) {
      case 'Low':
        return 'bg-emerald-100 text-emerald-800 border-emerald-200';
      case 'Medium':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'High':
        return 'bg-amber-100 text-amber-800 border-amber-200';
      default:
        return 'bg-slate-100 text-slate-800 border-slate-200';
    }
  };

  return (
    <div className="space-y-6" data-testid="predictions-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Predictions</h1>
          <p className="text-sm text-slate-600 leading-relaxed mt-2">
            View regression and classification predictions
          </p>
        </div>
        <button
          onClick={viewMetrics}
          data-testid="view-metrics-btn"
          className="btn-primary inline-flex items-center gap-2"
        >
          <Activity className="w-4 h-4" />
          View Metrics
        </button>
      </div>

      {/* Model Selection */}
      <div className="card p-6">
        <label className="block text-sm font-medium text-slate-700 mb-2">
          Select Regression Model
        </label>
        <select
          data-testid="model-select"
          value={modelType}
          onChange={(e) => setModelType(e.target.value)}
          className="w-full md:w-64 h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
        >
          <option value="linear_regression">Linear Regression</option>
          <option value="ridge_regression">Ridge Regression</option>
          <option value="mlp_regression">Neural Network (MLP)</option>
        </select>
      </div>

      {/* Regression Predictions */}
      {regressionData && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <TrendingUp className="w-5 h-5 text-slate-700" />
            <h3 className="text-xl font-medium text-slate-800">Regression Predictions</h3>
          </div>
          <p className="text-sm text-slate-600 mb-4">
            Model: <span className="font-medium">{modelType.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}</span>
            {' | '}
            Total Predictions: <span className="font-medium">{regressionData.count}</span>
          </p>
          
          {regressionChart && (
            <div className="chart-container">
              <Bar data={regressionChart} options={chartOptions} />
            </div>
          )}

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Average Cost</p>
              <p className="text-2xl font-bold text-slate-900">
                ${Math.round(regressionData.predictions.reduce((a, b) => a + b, 0) / regressionData.predictions.length).toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Min Cost</p>
              <p className="text-2xl font-bold text-slate-900">
                ${Math.round(Math.min(...regressionData.predictions)).toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg border border-slate-200">
              <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">Max Cost</p>
              <p className="text-2xl font-bold text-slate-900">
                ${Math.round(Math.max(...regressionData.predictions)).toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Classification Predictions */}
      {classificationData && (
        <div className="card p-6">
          <div className="flex items-center gap-2 mb-4">
            <PieChart className="w-5 h-5 text-slate-700" />
            <h3 className="text-xl font-medium text-slate-800">Classification Predictions</h3>
          </div>
          <p className="text-sm text-slate-600 mb-4">
            Total Predictions: <span className="font-medium">{classificationData.count}</span>
          </p>

          {/* Distribution */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {classificationData.distribution && Object.entries(classificationData.distribution).map(([category, count]) => (
              <div key={category} className={`p-6 rounded-lg border ${getCategoryColor(category)}`}>
                <p className="text-sm font-medium uppercase tracking-wider mb-2">{category} Cost</p>
                <p className="text-3xl font-bold">{count}</p>
                <p className="text-sm mt-1">
                  {((count / classificationData.count) * 100).toFixed(1)}% of total
                </p>
              </div>
            ))}
          </div>

          {classificationChart && (
            <div className="chart-container">
              <Bar data={classificationChart} options={chartOptions} />
            </div>
          )}

          {/* Sample Predictions */}
          <div className="mt-6">
            <h4 className="font-medium text-slate-700 mb-3">Sample Predictions</h4>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
              {classificationData.predictions.slice(0, 10).map((pred, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border text-center ${getCategoryColor(pred)}`}
                >
                  <p className="text-xs text-slate-500">Sample {idx + 1}</p>
                  <p className="font-medium">{pred}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Predictions;
