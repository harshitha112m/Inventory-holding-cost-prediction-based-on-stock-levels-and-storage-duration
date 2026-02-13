import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader2, TrendingUp, ArrowRight } from 'lucide-react';
import axios from 'axios';
import { Line, Bar, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const EDA = () => {
  const { dataId } = useParams();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [training, setTraining] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchEDA();
  }, [dataId]);

  const fetchEDA = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${API}/eda/${dataId}`);
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModels = async () => {
    try {
      setTraining(true);
      await axios.post(`${API}/train/${dataId}`);
      
      // Navigate to predictions page
      setTimeout(() => {
        navigate(`/predictions/${dataId}`);
      }, 1000);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setTraining(false);
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

  const distributionChart = data?.distributions?.holding_cost ? {
    labels: data.distributions.holding_cost.bins.slice(0, -1).map((b, i) => 
      `${Math.round(b)}-${Math.round(data.distributions.holding_cost.bins[i + 1])}`
    ),
    datasets: [{
      label: 'Frequency',
      data: data.distributions.holding_cost.values,
      backgroundColor: '#0EA5E9',
      borderColor: '#0F172A',
      borderWidth: 1
    }]
  } : null;

  return (
    <div className="space-y-6" data-testid="eda-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-slate-900">Exploratory Data Analysis</h1>
          <p className="text-sm text-slate-600 leading-relaxed mt-2">
            Visualize patterns and relationships in your inventory data
          </p>
        </div>
        <button
          onClick={handleTrainModels}
          data-testid="train-models-btn"
          disabled={training}
          className="btn-primary inline-flex items-center gap-2"
        >
          {training ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Training...
            </>
          ) : (
            <>
              Train Models
              <ArrowRight className="w-4 h-4" />
            </>
          )}
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card p-6 metric-card">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">Total Rows</p>
          <p className="text-3xl font-bold text-slate-900 tracking-tighter" data-testid="stat-total-rows">
            {data?.stats?.total_rows || 0}
          </p>
        </div>
        <div className="card p-6 metric-card">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">Total Columns</p>
          <p className="text-3xl font-bold text-slate-900 tracking-tighter">
            {data?.stats?.total_columns || 0}
          </p>
        </div>
        <div className="card p-6 metric-card">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">Numerical Features</p>
          <p className="text-3xl font-bold text-slate-900 tracking-tighter">
            {data?.stats?.numerical_features || 0}
          </p>
        </div>
        <div className="card p-6 metric-card">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-2">Categorical Features</p>
          <p className="text-3xl font-bold text-slate-900 tracking-tighter">
            {data?.stats?.categorical_features || 0}
          </p>
        </div>
      </div>

      {/* Distribution Chart */}
      {distributionChart && (
        <div className="card p-6">
          <h3 className="text-xl font-medium text-slate-800 mb-4">Holding Cost Distribution</h3>
          <div className="chart-container">
            <Bar data={distributionChart} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Scatter Plots */}
      {data?.scatter_plots && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {Object.entries(data.scatter_plots).map(([key, plotData]) => {
            const scatterData = {
              datasets: [{
                label: 'Data Points',
                data: plotData.x.map((x, i) => ({ x, y: plotData.y[i] })),
                backgroundColor: '#0EA5E9',
                borderColor: '#0F172A',
                borderWidth: 1,
                pointRadius: 4
              }]
            };

            return (
              <div key={key} className="card p-6">
                <h3 className="text-xl font-medium text-slate-800 mb-4">
                  {key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} vs Holding Cost
                </h3>
                <div className="chart-container">
                  <Scatter data={scatterData} options={chartOptions} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Box Plots Data */}
      {data?.box_plots && Object.keys(data.box_plots).length > 0 && (
        <div className="card p-6">
          <h3 className="text-xl font-medium text-slate-800 mb-4">Categorical Features Analysis</h3>
          <div className="space-y-4">
            {Object.entries(data.box_plots).map(([category, groups]) => (
              <div key={category}>
                <h4 className="font-medium text-slate-700 mb-2">
                  {category.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {Object.entries(groups).map(([group, values]) => (
                    <div key={group} className="p-3 bg-slate-50 rounded border border-slate-200">
                      <p className="text-sm font-medium text-slate-700">{group}</p>
                      <p className="text-xs text-slate-500">
                        Avg: ${Math.round(values.reduce((a, b) => a + b, 0) / values.length).toLocaleString()}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default EDA;
