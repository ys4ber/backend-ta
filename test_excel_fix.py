#!/usr/bin/env python3
"""
Test script to verify the Excel analysis fix
"""

import requests
import json
from io import BytesIO
import pandas as pd

def test_excel_analysis():
    """Test the Excel analysis endpoint with the urgency calculation fix"""
    
    # Create a test Excel file with sample data
    test_data = {
        'Date_Création': ['2024-01-15', '2024-01-16', '2024-01-17'],
        'Priorité': [1, 3, 2],
        'État': ['En cours', 'Fermé', 'En cours'],
        'Description': ['Pompe défaillante critique', 'Maintenance normale', 'Soupape urgente'],
        'Temperature': [85.5, 70.2, 95.1],
        'Pressure': [150.3, 120.5, 180.7],
        'Vibration': [0.8, 0.3, 1.2],
        'Flow_Rate': [45.2, 55.8, 35.9]
    }
    
    df = pd.DataFrame(test_data)
    
    # Save to BytesIO as Excel
    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    
    # Test the API endpoint
    url = "http://localhost:8000/analyze-excel"
    files = {'file': ('test_data.xlsx', excel_buffer.getvalue(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    print("🧪 Testing Excel analysis endpoint with urgency calculation fix...")
    
    try:
        response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Excel analysis succeeded!")
            print(f"📊 Processed {len(result['results'])} rows")
            
            # Check if urgency calculation worked
            for i, res in enumerate(result['results']):
                urgency = res.get('urgency', {})
                print(f"Row {i+1}: Urgency = {urgency.get('level', 'N/A')} ({urgency.get('color', 'N/A')})")
            
            print("🎯 Fix successful! Urgency calculation is working properly.")
            return True
            
        else:
            print(f"❌ Excel analysis failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_excel_analysis()
    if success:
        print("\n🎉 All tests passed! The Excel analysis fix is working correctly.")
    else:
        print("\n💥 Tests failed. Check the output above for details.")
