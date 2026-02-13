import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { TrendingUp, Database, Activity, ArrowRight } from 'lucide-react';

const Dashboard = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Data Upload',
      description: 'Upload CSV files, enter manual data, or use sample datasets',
      icon: Database,
      action: () => navigate('/upload'),
      color: 'bg-blue-500'
    },
    {
      title: 'Regression Models',
      description: 'Predict holding costs using Linear Regression, Ridge, and MLP',
      icon: TrendingUp,
      color: 'bg-emerald-500'
    },
    {
      title: 'Classification',
      description: 'Categorize costs into Low, Medium, and High risk categories',
      icon: Activity,
      color: 'bg-amber-500'
    }
  ];

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden rounded-lg bg-slate-900 p-8 md:p-12">
        <div className="relative z-10">
          <h1 className="text-4xl font-bold text-white mb-4">
            Inventory Holding Cost Prediction
          </h1>
          <p className="text-lg text-slate-300 mb-6 max-w-2xl">
            Leverage machine learning models to predict and categorize inventory holding costs
            based on stock levels, storage duration, and operational factors.
          </p>
          <button
            onClick={() => navigate('/upload')}
            data-testid="get-started-btn"
            className="btn-primary bg-sky-500 hover:bg-sky-600 inline-flex items-center gap-2"
          >
            Get Started
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>
        <div className="absolute top-0 right-0 w-1/2 h-full opacity-10">
          <img
            src="https://images.unsplash.com/photo-1761195696590-3490ea770aa1?crop=entropy&cs=srgb&fm=jpg&ixid=M3w3NDQ2NDJ8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjB3YXJlaG91c2UlMjBsb2dpc3RpY3MlMjBoaWdoJTIwdGVjaHxlbnwwfHx8fDE3NzA5NTY5NTd8MA&ixlib=rb-4.1.0&q=85"
            alt="Warehouse"
            className="w-full h-full object-cover"
          />
        </div>
      </div>

      {/* Features Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <div
              key={index}
              data-testid={`feature-card-${index}`}
              className="card p-6 cursor-pointer"
              onClick={feature.action}
            >
              <div className={`${feature.color} w-12 h-12 rounded-lg flex items-center justify-center mb-4`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-slate-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-sm text-slate-600 leading-relaxed">
                {feature.description}
              </p>
            </div>
          );
        })}
      </div>

      {/* Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card p-6">
          <h3 className="text-xl font-semibold text-slate-900 mb-4">Regression Models</h3>
          <ul className="space-y-3">
            <li className="flex items-start gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-2" />
              <div>
                <p className="font-medium text-slate-900">Linear Regression</p>
                <p className="text-sm text-slate-600">Baseline model with interpretable coefficients</p>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-2" />
              <div>
                <p className="font-medium text-slate-900">Ridge Regression</p>
                <p className="text-sm text-slate-600">L2 regularization to prevent overfitting</p>
              </div>
            </li>
            <li className="flex items-start gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-sky-500 mt-2" />
              <div>
                <p className="font-medium text-slate-900">Neural Network (MLP)</p>
                <p className="text-sm text-slate-600">Captures non-linear relationships</p>
              </div>
            </li>
          </ul>
        </div>

        <div className="card p-6">
          <h3 className="text-xl font-semibold text-slate-900 mb-4">Classification Categories</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg border border-emerald-200">
              <span className="font-medium text-emerald-800">Low Cost</span>
              <span className="text-sm text-emerald-600">&lt; $15,000</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg border border-blue-200">
              <span className="font-medium text-blue-800">Medium Cost</span>
              <span className="text-sm text-blue-600">$15,000 - $30,000</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-amber-50 rounded-lg border border-amber-200">
              <span className="font-medium text-amber-800">High Cost</span>
              <span className="text-sm text-amber-600">&gt; $30,000</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
