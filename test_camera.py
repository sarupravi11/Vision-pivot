#!/usr/bin/env python3
"""
Simple camera test script to verify camera access works
"""
import cv2
import os

# Set the environment variable to skip macOS camera authorization
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

def test_camera():
    print("Testing camera access...")
    
    # Try to open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to read frame from camera")
        cap.release()
        return False
    
    print("✅ Camera access successful!")
    print(f"Frame shape: {frame.shape}")
    
    # Clean up
    cap.release()
    return True

if __name__ == "__main__":
    success = test_camera()
    if success:
        print("\n🎉 Camera is working! You can now run the main application.")
    else:
        print("\n⚠️  Camera access failed. You may need to:")
        print("1. Grant camera permissions in System Preferences > Security & Privacy > Camera")
        print("2. Add Terminal (or your IDE) to the allowed apps list")
