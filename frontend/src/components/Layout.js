import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Upload, BarChart3, TrendingUp, Activity } from 'lucide-react';

const Layout = ({ children }) => {
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Data Upload', path: '/upload', icon: Upload },
  ];

  const isActive = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="p-6">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 bg-sky-500 rounded-lg flex items-center justify-center">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Inventory AI</h1>
              <p className="text-xs text-slate-400">Cost Prediction</p>
            </div>
          </div>

          <nav className="space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.path}
                  data-testid={`nav-${item.name.toLowerCase().replace(' ', '-')}`}
                  className={`sidebar-link flex items-center gap-3 px-4 py-3 rounded-md text-sm font-medium ${
                    isActive(item.path)
                      ? 'active text-white'
                      : 'text-slate-300 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-slate-800">
          <div className="text-xs text-slate-500">
            <p>Powered by ML Models</p>
            <p className="mt-1">Regression & Classification</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        <div className="bg-white/80 backdrop-blur-md border-b border-slate-200/50 sticky top-0 z-50">
          <div className="max-w-[1600px] mx-auto px-6 py-4">
            <h2 className="text-2xl font-semibold tracking-tight text-slate-900">
              Inventory Holding Cost Prediction
            </h2>
          </div>
        </div>
        <div className="max-w-[1600px] mx-auto p-6 md:p-8">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Layout;
