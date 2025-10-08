#!/bin/bash

# Fix for macOS camera authorization issue
export OPENCV_AVFOUNDATION_SKIP_AUTH=1

# Activate virtual environment
source .venv/bin/activate

echo "🚀 Starting Vision Pivot Attendance System..."
echo "📷 Camera optimized for MacBook"
echo "🔐 Login: admin / admin123"

# Run the application
python visionpivot.py
