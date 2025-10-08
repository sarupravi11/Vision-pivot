import cv2
import os
import face_recognition
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\SRIRAM\PycharmProjects\fd\face-attendance-system-921a2-firebase-adminsdk-ynaoj-d384c89ae0.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-attendance-system-921a2-default-rtdb.firebaseio.com/"
})


def load_encodings(encodings_folder='Encodings'):
    """
    Loads all the face encodings stored in the Encodings folder.
    """
    encodings = {}
    for file in os.listdir(encodings_folder):
        if file.endswith('.npy'):
            user_id = os.path.splitext(file)[0]  # Use user ID as the label
            encoding_path = os.path.join(encodings_folder, file)
            encodings[user_id] = np.load(encoding_path)
    return encodings


def update_firebase_attendance(self, user_id):
    """
    Updates the attendance data in the Firebase database.
    """
    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d %H:%M:%S')

    # Reference to the user's attendance data in Firebase
    user_ref = db.reference(f'Attendance system/{user_id}')
    user_data = user_ref.get()

    if user_data:
        # Increment total attendance
        total_attendance = user_data.get('total_attendance', 0) + 1

        # Update the last attendance time
        user_ref.update({
            'total_attendance': total_attendance,
            'last_attendance_time': date_time
        })
        print(f"Attendance updated for {user_data['name']} on {date_time}")
    else:
        print(f"No user found for user ID: {user_id}")



def mark_attendance(user_id):
    """
    Marks attendance by updating the Firebase database.
    """
    # Update the attendance in Firebase
    update_firebase_attendance(user_id)


def recognize_face_from_frame(frame, known_encodings, marked_ids):
    """
    Recognizes faces from the frame, compares them with known encodings,
    and draws a bounding box with the person's label around the detected face.
    """
    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Iterate through detected faces
    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        for user_id, stored_encoding in known_encodings.items():
            if user_id not in marked_ids:  # Check if the user ID has already been marked
                matches = face_recognition.compare_faces([stored_encoding], encoding)
                if matches[0]:
                    # Retrieve the user's name from Firebase for the user ID
                    user_data = db.reference(f'Attendance system/{user_id}').get()
                    if user_data:
                        name = user_data.get('name', user_id)
                        mark_attendance(user_id)
                        marked_ids.add(user_id)  # Add to the set of marked IDs

                        # Draw a bounding box around the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                        # Display the name label above the bounding box
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, name, (left, top - 10), font, 0.9, (255, 255, 255), 2)

                        print(f"Detected and marked: {name}")
                        return name
                    else:
                        print(f"User data not found for ID: {user_id}")
    return None


def main():
    """
    Main function to start the attendance system with live camera feed.
    """
    known_encodings = load_encodings()
    print("Encodings loaded. Starting camera...")

    # Set to keep track of IDs already marked during this session
    marked_ids = set()

    # Open the camera
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Set camera resolution (width, height)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB as face_recognition uses RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_name = recognize_face_from_frame(frame, known_encodings, marked_ids)

        # Display the frame with bounding boxes and labels
        cv2.imshow('Attendance System', frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
