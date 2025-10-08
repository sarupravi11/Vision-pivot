# Vision Pivot  
**Face-Based Attendance System with Real-Time Admin Dashboard**

![Firebase](https://img.shields.io/badge/Firebase-Backend-yellow)
![React](https://img.shields.io/badge/React-Admin%20Dashboard-blue)
![Python](https://img.shields.io/badge/Python-Face%20Recognition-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Project Summary

- **Title:** Vision Pivot  
- **Objective:** A secure, automated face recognition attendance system integrated with a React-based admin dashboard for real-time monitoring and management.

### Key Features

-  **Face Recognition:** Real-time attendance marking using MTCNN + FaceNet.
-  **Firebase Backend:** Central storage of attendance logs, user data, and admin authentication.
- ️ **React Admin Dashboard:**
  - Secure login for admins (Firebase Auth).
  - View, edit, and manage attendance logs.
  - Add/remove users and assign roles.
  - Export reports in CSV/PDF.
- 🛡 **Anti-Spoofing:** Prevents duplicate attendance for the same session.

---

##  Updated Workflow

### Pipeline Overview

1. **Admin Setup**
   - Secure login via React dashboard (Firebase Email/Password Auth).
   - Upload user images (e.g., `Images/user1/photo.jpg`) through the UI.

2. **Enrollment Phase**
   - `face_embeddings.py` automatically generates and stores `.npy` face embeddings for new users.

3. **Attendance Marking**
   - `attendance_system.py` runs a live camera feed, matches faces, and logs attendance in Firebase.

4. **Admin Actions**
   - View history, filter records, override or correct entries, and manage users/roles in real-time.

---

## Technology Stack

| Component             | Technologies Used                                               |
|----------------------|------------------------------------------------------------------|
| Face Recognition      | Python, OpenCV, MTCNN, FaceNet, face_recognition                |
| Backend               | Firebase (Realtime DB, Authentication)                          |
| Admin Dashboard       | React.js, Firebase SDK, Material-UI or Ant Design               |
| API Communication     | RESTful calls to Firebase                                       |
| Security              | Firebase Authentication with Role-Based Access                 |

---

## Key Files & Their Roles

| File                          | Purpose                                                      |
|-------------------------------|--------------------------------------------------------------|
| `face_embeddings.py`          | Generate face embeddings for enrolled users                 |
| `attendance_system.py`        | Capture camera feed, detect faces, and log to Firebase      |
| `vision-pivot-firebase-adminsdk.json` | Firebase service credentials for backend access     |
| `admin-dashboard/src/components/Login.js` | Handles admin authentication via Firebase        |
| `admin-dashboard/src/pages/Dashboard.js`  | View and manage attendance records                |
| `admin-dashboard/src/services/firebase.js`| Firebase API communication layer                 |

---

## Security & Authentication

-  **Firebase Auth** secures admin access via email/password.
-  Face embeddings are stored locally in `.npy` files (can be encrypted).
-  Role-based access for managing admin/staff permissions.

---

## ️ How to Run the Project

###  Prerequisites

#### Firebase Setup:
1. Enable Realtime Database and Email/Password Authentication in Firebase Console.
2. Download and place `vision-pivot-firebase-adminsdk.json` in the backend directory.

---

### Python Backend

```bash
pip install opencv-python face-recognition facenet-pytorch firebase-admin pyttsx3 pyqt5 pandas numpy yagmail
python attendance_system.py
```

---

###  React Admin Dashboard

```bash
cd admin-dashboard
npm install
npm start
```

Access the dashboard at: `http://localhost:3000`

---

##  Execution Flow

1. **Admin Login:**  
   Access `localhost:3000/login` and log in with Firebase credentials.

2. **Enroll Users:**  
   Upload a photo through the dashboard UI – `face_embeddings.py` generates encodings automatically.

3. **Mark Attendance:**  
   Run `attendance_system.py`, which detects faces and logs data in Firebase.

4. **Monitor & Export:**  
   Dashboard reflects real-time updates. Download CSV reports as needed.

---

## ✨Highlighted Code Snippets

### Admin Login (React)

```js
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";
const auth = getAuth();
signInWithEmailAndPassword(auth, email, password).then((userCredential) => {
  // Redirect to dashboard
});
```

---

### Log Attendance (Python)

```python
db.reference(f'attendance/{user_id}').update({
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'status': 'present'
})
```

---

## Demo Scenario

```json
{
  "user2": {
    "name": "Alice Smith",
    "last_attendance": "2024-05-20 15:30:00",
    "total_attendances": 3
  }
}
```

- Admin logs in with:
  - Email: `admin@visionpivot.com`
  - Password: `********`
- Enrolls user2 → Uploads image → encoding generated.
- Runs camera → user2 is marked present → Dashboard reflects update.
- Exports CSV of May 2024.

---

## Strengths & Innovations

- **Real-Time Firebase Sync:** Dashboard updates instantly via listeners.
- **No-Code Admin Interface:** Easily manage users and attendance.
- **Secure and Scalable:** Firebase-backed authentication and storage.

---

