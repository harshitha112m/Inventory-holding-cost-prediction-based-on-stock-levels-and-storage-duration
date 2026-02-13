import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import DataUpload from './pages/DataUpload';
import EDA from './pages/EDA';
import Predictions from './pages/Predictions';
import Metrics from './pages/Metrics';
import './App.css';

function App() {
  return (
    <div className="App">
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<DataUpload />} />
            <Route path="/eda/:dataId" element={<EDA />} />
            <Route path="/predictions/:dataId" element={<Predictions />} />
            <Route path="/metrics/:dataId" element={<Metrics />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </div>
  );
}

export default App;
