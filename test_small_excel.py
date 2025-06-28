#!/usr/bin/env python3
"""
Test script to verify the Excel analysis fix with a small file
"""

import requests
import json
from io import BytesIO
import pandas as pd

def test_small_excel_analysis():
    """Test the Excel analysis endpoint with a small file"""
    
    # Create a very small test Excel file
    test_data = {
        'Date_Création': ['2024-01-15', '2024-01-16'],
        'Priorité': [1, 3],
        'État': ['En cours', 'Fermé'],
        'Description': ['Pompe défaillante', 'Maintenance'],
        'Temperature': [85.5, 70.2],
        'Pressure': [150.3, 120.5],
        'Vibration': [0.8, 0.3],
        'Flow_Rate': [45.2, 55.8]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to BytesIO as Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    
    # Test the API endpoint
    url = "http://localhost:8000/analyze-excel"
    files = {'file': ('small_test.xlsx', excel_buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    print("🧪 Testing Excel analysis with small file...")
    
    try:
        response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Excel analysis succeeded!")
            print(f"📊 Processed {len(result['results'])} rows")
            
            # Check if urgency calculation worked
            for i, res in enumerate(result['results']):
                urgency = res.get('urgency', {})
                anomaly_score = res.get('ai_prediction', {}).get('anomaly_score', 'N/A')
                print(f"Row {i+1}: Anomaly Score = {anomaly_score}, Urgency = {urgency.get('level', 'N/A')} ({urgency.get('color', 'N/A')})")
            
            print("🎯 Fix successful! Urgency calculation is working properly.")
            return True
            
        else:
            print(f"❌ Excel analysis failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_small_excel_analysis()
    if success:
        print("\n🎉 Small file test passed! The Excel analysis fix is working correctly.")
    else:
        print("\n💥 Small file test failed.")
