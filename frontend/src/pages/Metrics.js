import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader2, ArrowLeft, TrendingUp, Target } from 'lucide-react';
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

const Metrics = () => {
  const { dataId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchMetrics();
  }, [dataId]);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/metrics/${dataId}`);
      setMetrics(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
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

  const r2ChartData = metrics?.regression_metrics ? {
    labels: Object.keys(metrics.regression_metrics),
    datasets: [{
      label: 'R² Score',
      data: Object.values(metrics.regression_metrics).map(m => m.r2_score),
      backgroundColor: '#0EA5E9',
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  const maeChartData = metrics?.regression_metrics ? {
    labels: Object.keys(metrics.regression_metrics),
    datasets: [{
      label: 'Mean Absolute Error (MAE)',
      data: Object.values(metrics.regression_metrics).map(m => m.mae),
      backgroundColor: '#F59E0B',
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  const rmseChartData = metrics?.regression_metrics ? {
    labels: Object.keys(metrics.regression_metrics),
    datasets: [{
      label: 'Root Mean Squared Error (RMSE)',
      data: Object.values(metrics.regression_metrics).map(m => m.rmse),
      backgroundColor: '#EF4444',
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  return (
    <div className="space-y-6" data-testid="metrics-page">
      <div className="flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate(`/predictions/${dataId}`)}
            className="inline-flex items-center gap-2 text-sm text-slate-600 hover:text-slate-900 mb-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Predictions
          </button>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Model Evaluation Metrics</h1>
          <p className="text-sm text-slate-600 leading-relaxed mt-2">
            Performance metrics for trained models
          </p>
        </div>
      </div>

      {/* Regression Metrics */}
      {metrics?.regression_metrics && (
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-slate-700" />
            <h2 className="text-2xl font-semibold tracking-tight text-slate-900">Regression Metrics</h2>
          </div>

          {/* Metrics Table */}
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full" data-testid="regression-metrics-table">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                      Model
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                      R² Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                      MAE
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                      RMSE
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.regression_metrics).map(([model, m], idx) => (
                    <tr key={idx} className="hover:bg-slate-50/80 transition-colors border-b border-slate-100 last:border-0">
                      <td className="px-6 py-4 text-sm font-medium text-slate-900">{model}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">{m.r2_score.toFixed(3)}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">${m.mae.toLocaleString()}</td>
                      <td className="px-6 py-4 text-sm text-slate-600">${m.rmse.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {r2ChartData && (
              <div className="card p-6">
                <h3 className="text-xl font-medium text-slate-800 mb-4">R² Score Comparison</h3>
                <div className="chart-container">
                  <Bar data={r2ChartData} options={chartOptions} />
                </div>
              </div>
            )}
            {maeChartData && (
              <div className="card p-6">
                <h3 className="text-xl font-medium text-slate-800 mb-4">MAE Comparison</h3>
                <div className="chart-container">
                  <Bar data={maeChartData} options={chartOptions} />
                </div>
              </div>
            )}
            {rmseChartData && (
              <div className="card p-6">
                <h3 className="text-xl font-medium text-slate-800 mb-4">RMSE Comparison</h3>
                <div className="chart-container">
                  <Bar data={rmseChartData} options={chartOptions} />
                </div>
              </div>
            )}
          </div>

          {/* Metrics Explanation */}
          <div className="card p-6 bg-slate-50">
            <h3 className="text-lg font-medium text-slate-800 mb-3">Understanding Metrics</h3>
            <div className="space-y-2 text-sm text-slate-600">
              <p><strong>R² Score:</strong> Proportion of variance explained by the model (higher is better, max 1.0)</p>
              <p><strong>MAE:</strong> Average absolute difference between predictions and actual values (lower is better)</p>
              <p><strong>RMSE:</strong> Root mean squared error, penalizes larger errors (lower is better)</p>
            </div>
          </div>
        </div>
      )}

      {/* Classification Metrics */}
      {metrics?.classification_metrics && (
        <div className="space-y-6">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-slate-700" />
            <h2 className="text-2xl font-semibold tracking-tight text-slate-900">Classification Metrics</h2>
          </div>

          <div className="card p-6">
            <h3 className="text-xl font-medium text-slate-800 mb-4">Class Distribution</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {metrics.classification_metrics.classes.map((cls, idx) => {
                const count = metrics.classification_metrics.distribution[cls];
                const total = Object.values(metrics.classification_metrics.distribution).reduce((a, b) => a + b, 0);
                const percentage = ((count / total) * 100).toFixed(1);

                const colors = {
                  'Low': 'bg-emerald-100 text-emerald-800 border-emerald-200',
                  'Medium': 'bg-blue-100 text-blue-800 border-blue-200',
                  'High': 'bg-amber-100 text-amber-800 border-amber-200'
                };

                return (
                  <div key={idx} className={`p-6 rounded-lg border ${colors[cls] || 'bg-slate-100 text-slate-800 border-slate-200'}`}>
                    <p className="text-sm font-medium uppercase tracking-wider mb-2">{cls} Cost</p>
                    <p className="text-3xl font-bold">{count}</p>
                    <p className="text-sm mt-1">{percentage}% of total</p>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Metrics;
