#!/usr/bin/env python3
"""
Script to set up admin credentials in Firestore database
"""
import firebase_admin
from firebase_admin import credentials, firestore

def setup_admin_credentials():
    """
    Creates admin collection and document in Firestore with default password
    """
    try:
        # Initialize Firebase (if not already initialized)
        try:
            cred = credentials.Certificate("vision-pivot-928a8-firebase-adminsdk-nbo5g-9402a11e5b.json")
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully")
        except ValueError:
            # App already initialized
            print("ℹ️  Firebase app already initialized")
        
        # Get Firestore client
        db = firestore.client()
        
        # Create admin collection and document
        admin_ref = db.collection('admin').document('admin')
        
        # Set admin password (you can change this)
        admin_data = {
            'password': 'admin123'  # Change this to your preferred password
        }
        
        admin_ref.set(admin_data)
        
        print("✅ Admin credentials created successfully!")
        print("📝 Login Details:")
        print("   Username: admin")
        print("   Password: admin123")
        print("\n🔒 You can change the password in Firebase Console later")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up admin credentials: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Setting up admin credentials in Firestore...")
    success = setup_admin_credentials()
    
    if success:
        print("\n🎉 Setup complete! You can now login to the app.")
    else:
        print("\n💡 If this fails, please set up manually via Firebase Console.")
