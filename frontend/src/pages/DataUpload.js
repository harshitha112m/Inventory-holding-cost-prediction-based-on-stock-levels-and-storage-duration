import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, FileSpreadsheet, Database, Loader2 } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const DataUpload = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [activeTab, setActiveTab] = useState('upload'); // upload, manual, sample

  // Manual entry form state
  const [formData, setFormData] = useState({
    stock_level_units: '',
    storage_duration_days: '',
    product_category: 'Electronics',
    product_value_usd: '',
    storage_type: 'Standard Warehouse',
    inventory_turnover: '',
    insurance_rate_percent: '',
    obsolescence_risk: 'Medium',
    storage_rent_usd_per_month: '',
    handling_cost_per_unit: '',
    security_level: 'Standard',
    seasonality: 'Non-Seasonal',
    supplier_reliability: 'Medium'
  });

  const onDrop = async (acceptedFiles) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setMessage('');

    try {
      const response = await axios.post(`${API}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setMessage(`Success! Uploaded ${response.data.rows} rows and ${response.data.columns} columns.`);
      
      // Navigate to EDA page after 2 seconds
      setTimeout(() => {
        navigate(`/eda/${response.data.id}`);
      }, 2000);
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    multiple: false
  });

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setMessage('');

    try {
      // Convert string values to numbers
      const payload = {
        ...formData,
        stock_level_units: parseFloat(formData.stock_level_units),
        storage_duration_days: parseFloat(formData.storage_duration_days),
        product_value_usd: parseFloat(formData.product_value_usd),
        inventory_turnover: parseFloat(formData.inventory_turnover),
        insurance_rate_percent: parseFloat(formData.insurance_rate_percent),
        storage_rent_usd_per_month: parseFloat(formData.storage_rent_usd_per_month),
        handling_cost_per_unit: parseFloat(formData.handling_cost_per_unit)
      };

      const response = await axios.post(`${API}/manual-entry`, payload);
      setMessage('Data entry saved successfully!');
      
      setTimeout(() => {
        navigate(`/eda/${response.data.id}`);
      }, 2000);
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleSampleData = async () => {
    setLoading(true);
    setMessage('');

    try {
      const response = await axios.get(`${API}/sample-data`);
      setMessage('Sample dataset loaded successfully!');
      
      setTimeout(() => {
        navigate(`/eda/${response.data.id}`);
      }, 2000);
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6" data-testid="data-upload-page">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-slate-900">Data Upload</h1>
        <p className="text-sm text-slate-600 leading-relaxed mt-2">
          Upload your inventory data, enter manually, or use a sample dataset
        </p>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-200">
        <button
          onClick={() => setActiveTab('upload')}
          data-testid="tab-upload"
          className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'upload'
              ? 'border-slate-900 text-slate-900'
              : 'border-transparent text-slate-600 hover:text-slate-900'
          }`}
        >
          <FileSpreadsheet className="w-4 h-4 inline mr-2" />
          Upload CSV
        </button>
        <button
          onClick={() => setActiveTab('manual')}
          data-testid="tab-manual"
          className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'manual'
              ? 'border-slate-900 text-slate-900'
              : 'border-transparent text-slate-600 hover:text-slate-900'
          }`}
        >
          <Upload className="w-4 h-4 inline mr-2" />
          Manual Entry
        </button>
        <button
          onClick={() => setActiveTab('sample')}
          data-testid="tab-sample"
          className={`px-4 py-2 font-medium text-sm border-b-2 transition-colors ${
            activeTab === 'sample'
              ? 'border-slate-900 text-slate-900'
              : 'border-transparent text-slate-600 hover:text-slate-900'
          }`}
        >
          <Database className="w-4 h-4 inline mr-2" />
          Sample Data
        </button>
      </div>

      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="card p-8">
          <div
            {...getRootProps()}
            data-testid="dropzone"
            className={`upload-zone ${
              isDragActive ? 'bg-sky-50 border-sky-400' : ''
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="w-12 h-12 mx-auto mb-4 text-slate-400" />
            {isDragActive ? (
              <p className="text-slate-700 font-medium">Drop the file here...</p>
            ) : (
              <div>
                <p className="text-slate-700 font-medium mb-2">
                  Drag & drop your CSV or Excel file here
                </p>
                <p className="text-sm text-slate-500">or click to browse</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Manual Entry Tab */}
      {activeTab === 'manual' && (
        <div className="card p-6">
          <form onSubmit={handleManualSubmit} className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Stock Level (units)
                </label>
                <input
                  type="number"
                  data-testid="input-stock-level"
                  value={formData.stock_level_units}
                  onChange={(e) => setFormData({ ...formData, stock_level_units: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Storage Duration (days)
                </label>
                <input
                  type="number"
                  data-testid="input-storage-duration"
                  value={formData.storage_duration_days}
                  onChange={(e) => setFormData({ ...formData, storage_duration_days: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Product Category
                </label>
                <select
                  data-testid="select-product-category"
                  value={formData.product_category}
                  onChange={(e) => setFormData({ ...formData, product_category: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                >
                  <option>Electronics</option>
                  <option>Automotive</option>
                  <option>Pharmaceuticals</option>
                  <option>Textiles</option>
                  <option>Food</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Product Value (USD)
                </label>
                <input
                  type="number"
                  step="0.01"
                  data-testid="input-product-value"
                  value={formData.product_value_usd}
                  onChange={(e) => setFormData({ ...formData, product_value_usd: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Storage Type
                </label>
                <select
                  data-testid="select-storage-type"
                  value={formData.storage_type}
                  onChange={(e) => setFormData({ ...formData, storage_type: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                >
                  <option>Standard Warehouse</option>
                  <option>Climate Controlled</option>
                  <option>Outdoor</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Inventory Turnover
                </label>
                <input
                  type="number"
                  step="0.01"
                  data-testid="input-inventory-turnover"
                  value={formData.inventory_turnover}
                  onChange={(e) => setFormData({ ...formData, inventory_turnover: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Insurance Rate (%)
                </label>
                <input
                  type="number"
                  step="0.01"
                  data-testid="input-insurance-rate"
                  value={formData.insurance_rate_percent}
                  onChange={(e) => setFormData({ ...formData, insurance_rate_percent: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Storage Rent (USD/month)
                </label>
                <input
                  type="number"
                  step="0.01"
                  data-testid="input-storage-rent"
                  value={formData.storage_rent_usd_per_month}
                  onChange={(e) => setFormData({ ...formData, storage_rent_usd_per_month: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">
                  Handling Cost (per unit)
                </label>
                <input
                  type="number"
                  step="0.01"
                  data-testid="input-handling-cost"
                  value={formData.handling_cost_per_unit}
                  onChange={(e) => setFormData({ ...formData, handling_cost_per_unit: e.target.value })}
                  className="w-full h-10 border-slate-200 focus:ring-2 focus:ring-slate-900/20 focus:border-slate-900 rounded-md bg-white px-3"
                  required
                />
              </div>
            </div>
            <button
              type="submit"
              data-testid="submit-manual-entry"
              disabled={loading}
              className="btn-primary w-full md:w-auto"
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
                  Processing...
                </>
              ) : (
                'Submit'
              )}
            </button>
          </form>
        </div>
      )}

      {/* Sample Data Tab */}
      {activeTab === 'sample' && (
        <div className="card p-8 text-center">
          <Database className="w-16 h-16 mx-auto mb-4 text-slate-400" />
          <h3 className="text-xl font-semibold text-slate-900 mb-2">Sample Dataset</h3>
          <p className="text-sm text-slate-600 mb-6">
            Load a pre-generated dataset with 100 samples to test the models
          </p>
          <button
            onClick={handleSampleData}
            data-testid="load-sample-data-btn"
            disabled={loading}
            className="btn-primary"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 inline animate-spin" />
                Loading...
              </>
            ) : (
              'Load Sample Data'
            )}
          </button>
        </div>
      )}

      {/* Message Display */}
      {message && (
        <div
          data-testid="upload-message"
          className={`p-4 rounded-lg ${
            message.includes('Error')
              ? 'bg-red-50 border border-red-200 text-red-800'
              : 'bg-emerald-50 border border-emerald-200 text-emerald-800'
          }`}
        >
          {message}
        </div>
      )}
    </div>
  );
};

export default DataUpload;
