@echo off
title Streamlit Dashboard

cd /d "%~dp0"

py --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed.
    echo Please install Python from https://www.python.org
    pause
    exit /b
)

py -m pip install streamlit pandas matplotlib seaborn openpyxl
py -m pip install scikit-learn

py -m streamlit run First_Page.py

pause
