#!/usr/bin/env python3
import requests
import sys
import json
import time
from datetime import datetime

class MLAPITester:
    def __init__(self, base_url="https://inventory-analytics-4.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.uploaded_data_id = None
        self.sample_data_id = None
        self.manual_data_id = None

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None):
        """Run a single API test"""
        url = f"{self.base_url}{endpoint}"
        headers = {'Content-Type': 'application/json'} if not files else {}

        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        print(f"URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files)
                else:
                    response = requests.post(url, json=data, headers=headers)
            
            print(f"Response Status: {response.status_code}")
            print(f"Response Headers: {dict(response.headers)}")
            
            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                return True, response.json() if response.text else {}
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                print(f"Response Text: {response.text}")
                return False, {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            return False, {}

    def test_root_endpoint(self):
        """Test root API endpoint"""
        success, response = self.run_test(
            "Root Endpoint",
            "GET",
            "/",
            200
        )
        return success

    def test_sample_data_loading(self):
        """Test sample data loading"""
        success, response = self.run_test(
            "Sample Data Loading",
            "GET",
            "/sample-data", 
            200
        )
        if success and 'id' in response:
            self.sample_data_id = response['id']
            print(f"Sample data ID: {self.sample_data_id}")
            print(f"Sample data rows: {response.get('rows', 0)}")
        return success

    def test_manual_entry(self):
        """Test manual data entry"""
        manual_data = {
            "stock_level_units": 5000.0,
            "storage_duration_days": 180.0,
            "product_category": "Electronics",
            "product_value_usd": 1500.0,
            "storage_type": "Climate Controlled",
            "inventory_turnover": 6.0,
            "insurance_rate_percent": 2.5,
            "obsolescence_risk": "Medium",
            "storage_rent_usd_per_month": 8.0,
            "handling_cost_per_unit": 2.5,
            "security_level": "High",
            "seasonality": "Non-Seasonal",
            "supplier_reliability": "High"
        }
        
        success, response = self.run_test(
            "Manual Data Entry",
            "POST",
            "/manual-entry",
            200,
            data=manual_data
        )
        if success and 'id' in response:
            self.manual_data_id = response['id']
            print(f"Manual data ID: {self.manual_data_id}")
        return success

    def test_eda_analysis(self, data_id, data_type):
        """Test EDA analysis for a given data ID"""
        if not data_id:
            print(f"❌ Skipping EDA test - no {data_type} data ID")
            return False
            
        success, response = self.run_test(
            f"EDA Analysis ({data_type})",
            "GET",
            f"/eda/{data_id}",
            200
        )
        
        if success:
            print(f"EDA Stats: {response.get('stats', {})}")
            print(f"Has distributions: {'distributions' in response}")
            print(f"Has scatter plots: {'scatter_plots' in response}")
        return success

    def test_model_training(self, data_id, data_type):
        """Test model training for a given data ID"""
        if not data_id:
            print(f"❌ Skipping training test - no {data_type} data ID")
            return False
            
        success, response = self.run_test(
            f"Model Training ({data_type})",
            "POST",
            f"/train/{data_id}",
            200
        )
        
        if success:
            print(f"Training metrics: {response.get('metrics', {})}")
        return success

    def test_regression_prediction(self, data_id, data_type, model_type="linear_regression"):
        """Test regression predictions"""
        if not data_id:
            print(f"❌ Skipping regression test - no {data_type} data ID")
            return False
            
        request_data = {
            "data_id": data_id,
            "model_type": model_type
        }
        
        success, response = self.run_test(
            f"Regression Prediction ({data_type}, {model_type})",
            "POST",
            f"/predict-regression/{data_id}",
            200,
            data=request_data
        )
        
        if success:
            print(f"Predictions count: {response.get('count', 0)}")
            print(f"Model type: {response.get('model_type', 'unknown')}")
        return success

    def test_classification_prediction(self, data_id, data_type):
        """Test classification predictions"""
        if not data_id:
            print(f"❌ Skipping classification test - no {data_type} data ID")
            return False
            
        success, response = self.run_test(
            f"Classification Prediction ({data_type})",
            "POST",
            f"/predict-classification/{data_id}",
            200
        )
        
        if success:
            print(f"Predictions count: {response.get('count', 0)}")
            print(f"Distribution: {response.get('distribution', {})}")
        return success

    def test_metrics_retrieval(self, data_id, data_type):
        """Test metrics retrieval"""
        if not data_id:
            print(f"❌ Skipping metrics test - no {data_type} data ID")
            return False
            
        success, response = self.run_test(
            f"Metrics Retrieval ({data_type})",
            "GET",
            f"/metrics/{data_id}",
            200
        )
        
        if success:
            print(f"Has regression metrics: {'regression_metrics' in response}")
            print(f"Has classification metrics: {'classification_metrics' in response}")
        return success

    def run_full_test_suite(self):
        """Run complete test suite"""
        print(f"🚀 Starting ML API Test Suite")
        print(f"Backend URL: {self.base_url}")
        print("=" * 60)
        
        # Test 1: Root endpoint
        self.test_root_endpoint()
        
        # Test 2: Sample data loading
        self.test_sample_data_loading()
        
        # Test 3: Manual data entry
        self.test_manual_entry()
        
        # Test 4: EDA Analysis (both sample and manual data)
        if self.sample_data_id:
            self.test_eda_analysis(self.sample_data_id, "sample")
        if self.manual_data_id:
            self.test_eda_analysis(self.manual_data_id, "manual")
        
        # Test 5: Model Training
        if self.sample_data_id:
            self.test_model_training(self.sample_data_id, "sample")
        
        # Test 6: Regression Predictions (all models)
        if self.sample_data_id:
            for model in ["linear_regression", "ridge_regression", "mlp_regression"]:
                self.test_regression_prediction(self.sample_data_id, "sample", model)
        
        # Test 7: Classification Predictions
        if self.sample_data_id:
            self.test_classification_prediction(self.sample_data_id, "sample")
        
        # Test 8: Metrics Retrieval
        if self.sample_data_id:
            self.test_metrics_retrieval(self.sample_data_id, "sample")

        # Print results
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        return self.tests_passed, self.tests_run

def main():
    tester = MLAPITester()
    passed, total = tester.run_full_test_suite()
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())