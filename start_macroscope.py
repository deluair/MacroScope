#!/usr/bin/env python3
"""
MacroScope Startup Script
Starts both the API server and Streamlit dashboard
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    
    print("🚀 Starting MacroScope Economic Intelligence Platform...")
    print("="*60)
    
    # Check if synthetic data exists, generate if not
    data_files = [
        "primary_indicators.csv",
        "regional_economic.csv", 
        "financial_markets.csv",
        "international_trade.csv",
        "alternative_indicators.csv"
    ]
    
    synthetic_dir = project_root / "data" / "synthetic"
    missing_files = [f for f in data_files if not (synthetic_dir / f).exists()]
    
    if missing_files:
        print("📊 Generating missing synthetic data...")
        subprocess.run([sys.executable, "generate_data.py"], cwd=project_root)
        print("✅ Synthetic data generated successfully!")
    
    print("\n🔧 Starting services...")
    
    # Start API server
    print("\n📡 Starting API Server on http://localhost:8000")
    api_cmd = [
        sys.executable, "-m", "uvicorn", 
        "src.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    # Start Streamlit dashboard
    print("📊 Starting Dashboard on http://localhost:8501")
    dashboard_cmd = [
        "streamlit", "run", 
        "src/dashboard/streamlit_app.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    
    print("\n" + "="*60)
    print("🎉 MacroScope is ready!")
    print("\n📡 API Documentation: http://localhost:8000/docs")
    print("📊 Dashboard: http://localhost:8501")
    print("\nPress Ctrl+C to stop all services")
    print("="*60)
    
    try:
        # Start both processes
        api_process = subprocess.Popen(api_cmd, cwd=project_root)
        time.sleep(2)  # Give API time to start
        dashboard_process = subprocess.Popen(dashboard_cmd, cwd=project_root)
        
        # Wait for both processes
        api_process.wait()
        dashboard_process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutting down MacroScope...")
        try:
            api_process.terminate()
            dashboard_process.terminate()
        except:
            pass
        print("✅ MacroScope stopped successfully!")

if __name__ == "__main__":
    main()