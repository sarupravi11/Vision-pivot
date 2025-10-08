#!/usr/bin/env python3
"""
Script to set up Firebase Realtime Database structure for attendance system
"""
import firebase_admin
from firebase_admin import credentials, db
import json

def setup_realtime_database():
    """
    Creates the required structure in Firebase Realtime Database
    """
    try:
        # Initialize Firebase (if not already initialized)
        try:
            cred = credentials.Certificate("vision-pivot-928a8-firebase-adminsdk-nbo5g-9402a11e5b.json")
            firebase_admin.initialize_app(cred, {
                'databaseURL': "https://vision-pivot-928a8-default-rtdb.firebaseio.com/"
            })
            print("✅ Firebase initialized successfully")
        except ValueError:
            # App already initialized
            print("ℹ️  Firebase app already initialized")
        
        # Get Realtime Database reference
        ref = db.reference()
        
        # Create basic structure for attendance system
        initial_data = {
            "Attendance system": {
                "README": {
                    "description": "This is the attendance system database",
                    "created": "2025-10-03",
                    "structure": "Each user will have their own node with attendance data"
                }
            }
        }
        
        # Check if structure already exists
        existing_data = ref.get()
        if existing_data and "Attendance system" in existing_data:
            print("ℹ️  Attendance system structure already exists")
        else:
            # Set the initial structure
            ref.update(initial_data)
            print("✅ Realtime Database structure created successfully!")
        
        print("\n📊 Database Structure:")
        print("   └── Attendance system/")
        print("       └── [User IDs will be added here when members are registered]")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up Realtime Database: {e}")
        return False

def test_database_connection():
    """
    Test if we can read/write to the database
    """
    try:
        ref = db.reference("Attendance system")
        test_data = ref.get()
        print(f"✅ Database connection test successful")
        print(f"📊 Current data: {test_data}")
        return True
    except Exception as e:
        print(f"❌ Database connection test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up Firebase Realtime Database structure...")
    
    success = setup_realtime_database()
    
    if success:
        print("\n🔍 Testing database connection...")
        test_success = test_database_connection()
        
        if test_success:
            print("\n🎉 Setup complete! The attendance system should now work.")
            print("\n💡 Next steps:")
            print("   1. Run the app: ./run_app.sh")
            print("   2. Login with: admin / admin123")
            print("   3. Add members through the admin panel")
        else:
            print("\n⚠️  Setup completed but connection test failed.")
    else:
        print("\n💥 Setup failed. Please check your Firebase configuration.")
