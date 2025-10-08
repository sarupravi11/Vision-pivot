import shutil
import cv2
import threading
import firebase_admin
from firebase_admin import credentials, db,firestore
import face_embeddings
import pyttsx3
from facenet_pytorch import InceptionResnetV1,MTCNN
import torch
import os
from datetime import datetime
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QLabel, QLineEdit,QVBoxLayout, QSpacerItem, QSizePolicy, QFrame, QWidget,QMessageBox,QGridLayout,QApplication,QFileDialog,QHBoxLayout,
                             QPushButton, QDateEdit, QDialog, QDialogButtonBox,QComboBox
)
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap, QPalette, QBrush, QFont
from PyQt5.QtCore import Qt,QDate
import sys
import yagmail


# Initialize Firebase
cred = credentials.Certificate(
    r"vision-pivot-928a8-firebase-adminsdk-nbo5g-9402a11e5b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://vision-pivot-928a8-default-rtdb.firebaseio.com/"
})
# Get database references
db1 = firestore.client()  # For admin login
# Note: db is imported from firebase_admin.db for Realtime Database
# Initialize the attendance file path
attendance_file = 'Attendance/attendance.csv'
if not os.path.exists('Attendance'):
    os.makedirs('Attendance')
# Initialize attendance data
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=['Name', 'Date', 'Entry Time', 'Exit Time', 'User ID'])
    df.to_csv(attendance_file, index=False)

class BaseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.set_background_image()

    def set_background_image(self):
        oImage = QPixmap("gradient-tool-example1.jpg")
        sImage = oImage.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(sImage))
        self.setPalette(palette)

    def resizeEvent(self, event):
        self.set_background_image()
        super().resizeEvent(event)
class AttendanceApp(BaseWindow):
    def __init__(self):
        super().__init__()
        self.root_layout = QVBoxLayout(self)
        self.engine = pyttsx3.init()
        # Store recognized people and marked IDs
        self.recognized_people = set()
        self.marked_ids = set()
        self.email_sent = {}

        # Initialize page management stack
        self.pages = []

        # Load face encodings (stored as .pt files)
        self.known_encodings = []
        self.engine_thread = threading.Thread(target=self.init_engine)
        self.engine_thread.start()

        self.entry_camera_thread = None
        self.exit_camera_thread = None
        self.camera_running = False
        self.marked_entry_ids = set()  # Track users marked for entry attendance
        self.marked_exit_ids = set()  # Track users marked for exit attendance
        self.attendance_file = 'attendance.csv'  # Path to attendance CSV file
        self.entry_camera_source = 0  # Default to Webcam
        self.exit_camera_source = 0  # Use same camera for both (MacBook has only one built-in camera)

        # Initialize UI components without loading heavy resources
        self.entry_camera_feed  = None
        self.exit_camera_feed = None

        self.mode = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.mtcnn = MTCNN(keep_all=True, device=self.device,min_face_size=40)
        # Initialize home page (or any other UI page setup)
        self.home_page()

    def home_page(self):
        self.clear_frame()
        self.setWindowState(Qt.WindowMaximized)

        self.known_encodings = self.load_encodings('Encodings')

        # Existing frame for the entire layout
        frame = QFrame()
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(1632, 918)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Main layout to fill available space
        main_layout = QVBoxLayout(frame)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Admin button at the top left corner
        self.admin_button = QPushButton("Admin")
        self.admin_button.setFixedSize(100, 40)
        self.admin_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(5, 161, 88, 0.8); 
                color: white; 
                border-radius: 15px;  
                font: bold 14px 'Segoe UI';  
                padding: 8px;  
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(9, 140, 78, 0.8); 
            }
            QPushButton:pressed {
                background-color: rgba(5, 109, 60, 0.8);
            }
        """)
        self.admin_button.clicked.connect(self.login_page)
        main_layout.addWidget(self.admin_button, alignment=Qt.AlignTop | Qt.AlignLeft)

        # Add some spacing between the admin button and the camera layout
        main_layout.addStretch(1)

        # Camera layouts for Entry and Exit at the center
        camera_layout = QHBoxLayout()
        camera_layout.setSpacing(50)
        # Add stretch to center the camera feeds
        camera_layout.addStretch(1)

        # Entry camera section
        entry_camera_layout = QVBoxLayout()
        entry_label = QLabel("Entry Camera")
        entry_label.setStyleSheet("color: #343A40; font: bold 18px 'Segoe UI';background: transparent;")
        entry_label.setAlignment(Qt.AlignCenter)
        entry_camera_layout.addWidget(entry_label)

        self.entry_camera_feed = QLabel()
        self.entry_camera_feed.setFixedSize(480, 360)  # 4:3 aspect ratio for MacBook camera
        self.entry_camera_feed.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7); 
            border-radius: 15px;
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
        """)
        entry_camera_layout.addWidget(self.entry_camera_feed)

        # Dropdown menu for Entry camera selection
        self.entry_camera_dropdown = QComboBox()
        self.entry_camera_dropdown.addItems(["Built-in Camera", "External Camera (if available)"])
        self.entry_camera_dropdown.setStyleSheet("""
            QComboBox {
                font: 14px 'Segoe UI';
                padding: 5px;
                border: 1px solid rgba(0, 0, 0, 0.3);
                border-radius: 10px;
            }
        """)
        self.entry_camera_dropdown.currentIndexChanged.connect(self.update_entry_camera_source)
        entry_camera_layout.addWidget(self.entry_camera_dropdown)

        # Exit camera section
        exit_camera_layout = QVBoxLayout()
        exit_label = QLabel("Exit Camera")
        exit_label.setStyleSheet("color: #343A40; font: bold 18px 'Segoe UI';background: transparent;")
        exit_label.setAlignment(Qt.AlignCenter)
        exit_camera_layout.addWidget(exit_label)

        self.exit_camera_feed = QLabel()
        self.exit_camera_feed.setFixedSize(480, 360)  # 4:3 aspect ratio for MacBook camera
        self.exit_camera_feed.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7); 
            border-radius: 15px;
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
        """)
        exit_camera_layout.addWidget(self.exit_camera_feed)

        # Dropdown menu for Exit camera selection
        self.exit_camera_dropdown = QComboBox()
        self.exit_camera_dropdown.addItems(["Built-in Camera (Shared)", "External Camera (if available)"])
        self.exit_camera_dropdown.setStyleSheet("""
            QComboBox {
                font: 14px 'Segoe UI';
                padding: 5px;
                border: 1px solid rgba(0, 0, 0, 0.3);
                border-radius: 10px;
            }
        """)
        self.exit_camera_dropdown.currentIndexChanged.connect(self.update_exit_camera_source)
        exit_camera_layout.addWidget(self.exit_camera_dropdown)

        # Add entry and exit camera layouts to the main camera layout
        camera_layout.addLayout(entry_camera_layout)
        camera_layout.addLayout(exit_camera_layout)
        # Add stretch to ensure they stay centered
        camera_layout.addStretch(1)
        # Add the camera layout directly to the main layout
        main_layout.addLayout(camera_layout)

        # Add some spacing between the camera feeds and the close button
        main_layout.addStretch(1)

        # Close button at the bottom middle
        close_button = QPushButton("Close")
        close_button.setFixedSize(120, 40)
        close_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(205, 92, 92, 0.9); 
                color: white; 
                border-radius: 10px;  
                font: bold 16px 'Segoe UI';  
                padding: 8px;  
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9); 
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }
        """)
        close_button.clicked.connect(self.close_app)
        main_layout.addWidget(close_button, alignment=Qt.AlignBottom | Qt.AlignCenter)

        # Add the frame to the root layout and make it fill available space
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.start_video_streams()

    def login_page(self):
        self.stop_video()  # Ensure video threads stop
        self.clear_frame()
        self.setWindowState(Qt.WindowMaximized)
        # Hide the Admin button while on the login page
        if hasattr(self, 'admin_button'):
            self.admin_button.setVisible(False)

        # Login frame
        frame = QFrame()
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(500, 500)

        # Form layout for login
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(40, 50, 40, 50)
        form_layout.setSpacing(20)

        # Title label
        title_label = QLabel("Log into your Account",frame)
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title_label)

        # Spacer after title
        spacer_after_title = QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding)
        form_layout.addItem(spacer_after_title)

        # Username entry
        self.username_entry = QLineEdit(frame)
        self.username_entry.setPlaceholderText("Username")
        self.username_entry.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 0.7); 
            color: #343A40; 
            font: 14px 'Segoe UI'; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
            """
        )
        self.username_entry.setFixedHeight(50)
        self.username_entry.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.username_entry)

        # Password entry
        self.password_entry = QLineEdit(frame)
        self.password_entry.setPlaceholderText("Password")
        self.password_entry.setEchoMode(QLineEdit.Password)
        self.password_entry.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 0.7); 
            color: #343A40; 
            font: 14px 'Segoe UI'; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
            """
        )
        self.password_entry.setFixedHeight(50)
        self.password_entry.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.password_entry)

        # Spacer to increase gap between password and login button
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        form_layout.addItem(spacer)

        # Login button
        login_button = QPushButton("Login",frame)
        login_button.setFixedSize(120, 40)
        login_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
            """
        )
        login_button.clicked.connect(self.validate_login)
        form_layout.addWidget(login_button, alignment=Qt.AlignCenter)

        # Back button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }
            """
        )
        back_button.clicked.connect(self.home_page)
        form_layout.addWidget(back_button, alignment=Qt.AlignCenter)
        # Re-add the frame to the layout, making sure it's always centered
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

        # Update the size and position of the frame
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def admin_panel_page(self):
        self.clear_frame()  # Clear any existing UI elements
        self.setWindowState(Qt.WindowMaximized)
        # Create admin panel frame
        frame = QFrame()
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(500, 500)

        # Form layout for admin panel
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(40, 50, 40, 50)
        form_layout.setSpacing(20)

        # Title label
        title_label = QLabel("Admin Panel")
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title_label)

        # Spacer for aesthetics
        form_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Manage Members Button
        manage_members_button = QPushButton("Manage Members")
        manage_members_button.setFixedSize(240, 50)
        manage_members_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        manage_members_button.clicked.connect(self.manage_members_page)
        form_layout.addWidget(manage_members_button, alignment=Qt.AlignCenter)

        # Show Voice Message Button
        show_voice_message_button = QPushButton("Show Voice Message")
        show_voice_message_button.setFixedSize(240, 50)
        show_voice_message_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        show_voice_message_button.clicked.connect(self.show_voice_message_page)
        form_layout.addWidget(show_voice_message_button, alignment=Qt.AlignCenter)

        # Download Data Button
        download_data_button = QPushButton("Download Data")
        download_data_button.setFixedSize(240, 50)
        download_data_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        download_data_button.clicked.connect(self.download_data_page)
        form_layout.addWidget(download_data_button, alignment=Qt.AlignCenter)

        # Spacer to balance layout
        form_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Back Button
        back_button = QPushButton("Back")
        back_button.setFixedSize(240, 50)
        back_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }
        """)
        back_button.clicked.connect(self.home_page)
        form_layout.addWidget(back_button, alignment=Qt.AlignCenter)

        # Re-add the frame to the layout, making sure it's always centered
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

        # Update the size and position of the frame
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def manage_members_page(self):
        self.clear_frame()  # Clears existing UI elements
        self.setWindowState(Qt.WindowMaximized)
        # Create manage members frame
        frame = QFrame()
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(600, 440)

        # Set layout for frame
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        # Title label
        title_label = QLabel("Manage Members")
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Add Member button
        add_member_button = QPushButton("Add Member")
        add_member_button.setFixedSize(220, 50)
        add_member_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        add_member_button.clicked.connect(self.add_member_page)
        layout.addWidget(add_member_button, alignment=Qt.AlignCenter)

        # Remove Member button
        remove_member_button = QPushButton("Remove Member")
        remove_member_button.setFixedSize(220, 50)
        remove_member_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        remove_member_button.clicked.connect(self.remove_member_page)
        layout.addWidget(remove_member_button, alignment=Qt.AlignCenter)

        # View Member button
        view_member_button = QPushButton("View Member")
        view_member_button.setFixedSize(220, 50)
        view_member_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        view_member_button.clicked.connect(self.view_member_page)
        layout.addWidget(view_member_button, alignment=Qt.AlignCenter)

        # Update Data button
        update_data_button = QPushButton("Update Data")
        update_data_button.setFixedSize(220, 50)
        update_data_button.setStyleSheet("""
           QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
        """)
        update_data_button.clicked.connect(self.manage_data_page)
        layout.addWidget(update_data_button, alignment=Qt.AlignCenter)

        # Back button
        back_button = QPushButton("Back")
        back_button.setFixedSize(220, 50)
        back_button.setStyleSheet("""
          QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }
        """)
        back_button.clicked.connect(self.admin_panel_page)
        layout.addWidget(back_button, alignment=Qt.AlignCenter)

        # Spacer for layout balance
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def add_member_page(self):
        # Clear the current frame
        self.clear_frame()
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowTitle("Add Member")
        self.setMinimumSize(800, 600)

        # Create a frame for the add member form
        frame = QFrame(self)
        frame.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
            """
        )
        frame.setFixedSize(450, 650)  # Enlarged frame size

        # Layout for the form
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(50, 50, 50, 50)
        form_layout.setSpacing(30)

        # Title label
        title_label = QLabel("Add Member", frame)
        title_label.setStyleSheet("color: white; font: bold 28px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)

        # Upload Button
        self.upload_button = QPushButton("Upload Image Dataset", frame)
        self.upload_button.setStyleSheet(
            """
            background-color: rgba(255, 255, 255, 0.7);
            color: #343A40;
            font: 14px 'Segoe UI';
            padding: 12px;
            border-radius: 10px;
            """
        )
        self.upload_button.clicked.connect(self.upload_image_dataset)

        # Folder label to show the uploaded path
        self.upload_folder_path = QLabel("", frame)
        self.upload_folder_path.setStyleSheet("color: white; font: 12px 'Arial'; background:transparent;")
        self.upload_folder_path.setAlignment(Qt.AlignCenter)

        # Define a common style for input fields
        field_style = """
            background-color: rgba(255, 255, 255, 0.7);
            color: #343A40;
            font: 14px 'Segoe UI';
            padding: 12px;
            border-radius: 10px;
        """

        # Employee ID Field
        self.employee_id_entry = QLineEdit(frame)
        self.employee_id_entry.setPlaceholderText("Employee ID")
        self.employee_id_entry.setFixedHeight(50)  # Enlarged height
        self.employee_id_entry.setStyleSheet(field_style)

        # Name Field
        self.name_entry = QLineEdit(frame)
        self.name_entry.setPlaceholderText("Name")
        self.name_entry.setFixedHeight(50)  # Enlarged height
        self.name_entry.setStyleSheet(field_style)

        # Email Field (new)
        self.email_entry = QLineEdit(frame)
        self.email_entry.setPlaceholderText("Email")
        self.email_entry.setFixedHeight(50)  # Enlarged height
        self.email_entry.setStyleSheet(field_style)

        # Role Field
        self.role_entry = QLineEdit(frame)
        self.role_entry.setPlaceholderText("Role")
        self.role_entry.setFixedHeight(50)  # Enlarged height
        self.role_entry.setStyleSheet(field_style)

        # Add Member Button
        add_member_button = QPushButton("Add Member", frame)
        add_member_button.setFixedSize(200, 40)  # Set a consistent size for the button
        add_member_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(5, 161, 88, 0.8); 
                color: white; 
                border-radius: 10px;  
                font: bold 14px 'Segoe UI';  
                padding: 8px;  
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(9, 140, 78, 0.8); 
            }
            QPushButton:pressed {
                background-color: rgba(5, 109, 60, 0.8);
            }
            """
        )

        add_member_button.clicked.connect(self.add_member)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(200, 40)  # Match the size of Add Member Button
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 14px 'Segoe UI';  
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)

        # Add widgets to form layout
        form_layout.addWidget(title_label)
        form_layout.addWidget(self.upload_button)
        form_layout.addWidget(self.upload_folder_path)
        form_layout.addWidget(self.employee_id_entry)
        form_layout.addWidget(self.name_entry)
        form_layout.addWidget(self.email_entry)  # Add email field here
        form_layout.addWidget(self.role_entry)
        form_layout.addWidget(add_member_button,alignment=Qt.AlignCenter)
        form_layout.addWidget(back_button, alignment=Qt.AlignCenter)

        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def remove_member_page(self):
        # Clear the current frame
        self.clear_frame()

        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(800, 600)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Form frame setup
        frame = QFrame(self)
        frame.setStyleSheet("""
                                    background-color: rgba(255, 255, 255, 0.2);
                                    border-radius: 20px;
                                    box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                                                -8px -8px 20px rgba(255, 255, 255, 0.9);
                                    """)
        frame.setFixedSize(320, 280)
        self.main_layout.addWidget(frame)

        # Form layout within the frame
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(20)

        # Title label
        title_label = QLabel("View Member", frame)
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title_label)

        # Employee ID Field
        self.remove_id_entry = QLineEdit(frame)
        self.remove_id_entry.setPlaceholderText("Employee ID")
        self.remove_id_entry.setFixedHeight(30)
        self.remove_id_entry.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.7); color: #343A40; font: 12px 'Arial'; padding: 8px; border-radius: 10px;"
        )
        form_layout.addWidget(self.remove_id_entry)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # OK Button
        ok_button = QPushButton("OK", frame)
        ok_button.setFixedSize(120, 40)
        ok_button.setStyleSheet(
            """
                        QPushButton {
                            background-color: rgba(5, 161, 88, 0.8); 
                            color: white; 
                            border-radius: 10px;  
                            font: bold 14px 'Segoe UI';  
                            padding: 8px;  
                            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                                        -4px -4px 10px rgba(255, 255, 255, 0.7);
                        }
                        QPushButton:hover {
                            background-color: rgba(9, 140, 78, 0.8); 
                        }
                        QPushButton:pressed {
                            background-color: rgba(5, 109, 60, 0.8);
                        }
                    """
        )
        ok_button.clicked.connect(self.show_member_details_remove)
        button_layout.addWidget(ok_button)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Add button layout to the form layout and center it
        button_layout.setAlignment(Qt.AlignCenter)
        form_layout.addLayout(button_layout)
        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def view_member_page(self):
        # Clear the current frame
        self.clear_frame()

        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(800, 600)



        # Form frame setup
        frame = QFrame(self)
        frame.setStyleSheet("""
                            background-color: rgba(255, 255, 255, 0.2);
                            border-radius: 20px;
                            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                                        -8px -8px 20px rgba(255, 255, 255, 0.9);
                            """)
        frame.setFixedSize(320, 280)


        # Form layout within the frame
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(20)

        # Title label
        title_label = QLabel("View Member", frame)
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title_label)

        # Employee ID Field
        self.view_id_entry = QLineEdit(frame)
        self.view_id_entry.setPlaceholderText("Employee ID")
        self.view_id_entry.setFixedHeight(30)
        self.view_id_entry.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.7); color: #343A40; font: 12px 'Arial'; padding: 8px; border-radius: 10px;"
        )
        form_layout.addWidget(self.view_id_entry)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # OK Button
        ok_button = QPushButton("OK", frame)
        ok_button.setFixedSize(120, 40)
        ok_button.setStyleSheet(
            """
                        QPushButton {
                            background-color: rgba(5, 161, 88, 0.8); 
                            color: white; 
                            border-radius: 10px;  
                            font: bold 14px 'Segoe UI';  
                            padding: 8px;  
                            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                                        -4px -4px 10px rgba(255, 255, 255, 0.7);
                        }
                        QPushButton:hover {
                            background-color: rgba(9, 140, 78, 0.8); 
                        }
                        QPushButton:pressed {
                            background-color: rgba(5, 109, 60, 0.8);
                        }
                    """
        )
        ok_button.clicked.connect(self.show_member_details_view)
        button_layout.addWidget(ok_button)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Add button layout to the form layout and center it
        button_layout.setAlignment(Qt.AlignCenter)
        form_layout.addLayout(button_layout)
        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)
    def manage_data_page(self):
        # Clear the current frame
        self.clear_frame()

        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(800, 600)

        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)

        # Form frame setup
        frame = QFrame(self)
        frame.setStyleSheet("""
                    background-color: rgba(255, 255, 255, 0.2);
                    border-radius: 20px;
                    box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                                -8px -8px 20px rgba(255, 255, 255, 0.9);
                    """)
        frame.setFixedSize(320, 280)
        self.main_layout.addWidget(frame)

        # Form layout within the frame
        form_layout = QVBoxLayout(frame)
        form_layout.setContentsMargins(40, 40, 40, 40)
        form_layout.setSpacing(20)

        # Title label
        title_label = QLabel("Update Data",frame)
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(title_label)

        # Employee ID Field
        self.data_id_entry = QLineEdit(frame)
        self.data_id_entry.setPlaceholderText("Employee ID")
        self.data_id_entry.setFixedHeight(30)
        self.data_id_entry.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.7); color: #343A40; font: 12px 'Segoe UI'; padding: 8px; border-radius: 10px;"
        )
        form_layout.addWidget(self.data_id_entry)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        # OK Button
        ok_button = QPushButton("OK",frame)
        ok_button.setFixedSize(120, 40)
        ok_button.setStyleSheet(
            """
                        QPushButton {
                            background-color: rgba(5, 161, 88, 0.8); 
                            color: white; 
                            border-radius: 10px;  
                            font: bold 14px 'Segoe UI';  
                            padding: 8px;  
                            box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                                        -4px -4px 10px rgba(255, 255, 255, 0.7);
                        }
                        QPushButton:hover {
                            background-color: rgba(9, 140, 78, 0.8); 
                        }
                        QPushButton:pressed {
                            background-color: rgba(5, 109, 60, 0.8);
                        }
                    """
        )
        ok_button.clicked.connect(self.view_and_edit_employee)
        button_layout.addWidget(ok_button)

        # Back Button
        back_button = QPushButton("Back",frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Add button layout to the form layout and center it
        button_layout.setAlignment(Qt.AlignCenter)
        form_layout.addLayout(button_layout)
        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)


    def show_voice_message_page(self):
        pass

    def validate_login(self):
        username = self.username_entry.text()
        password = self.password_entry.text()

        if not username or not password:
            self.show_custom_warning("Input Error", "Username or password cannot be empty.", "warning")
            return

        try:
            user_ref = db1.collection('admin').document('admin')
            user_data = user_ref.get()

            if user_data.exists:
                stored_data = user_data.to_dict()
                stored_username = 'admin'
                stored_password = stored_data.get('password')

                if username == stored_username and password == stored_password:
                    # Show success message
                    self.show_custom_warning("Login Successful", "You have successfully logged in.", "info")

                    # Switch to admin panel
                    self.admin_panel_page()
                else:
                    self.show_custom_warning("Login Failed", "Invalid credentials.", "warning")
            else:
                self.show_custom_warning("Login Failed", "User not found.", "warning")

        except Exception as e:
            self.show_custom_warning("Error", f"Error accessing Firestore: {e}", "error")


    def upload_image_dataset(self):
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder_path:
                self.upload_folder_path.setText(folder_path)
                self.folder_path = folder_path

        # Assuming you have Firebase already set up

    def add_member(self):
        employee_id = self.employee_id_entry.text()
        name = self.name_entry.text()
        role = self.role_entry.text()
        email = self.email_entry.text()  # Assuming you have an email input field named email_entry

        # Check if all required fields and folder path are set
        if not all([employee_id, name, role, email, hasattr(self, 'folder_path')]):
            self.show_custom_warning("Input Error", "Please complete all fields and upload the image dataset.",
                                     "warning")
            return

        # Create the directory in 'Images' folder if it doesn't exist
        person_folder = os.path.join('Images', employee_id)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)

        # Move images to the 'Images' folder
        for image_name in os.listdir(self.folder_path):
            image_path = os.path.join(self.folder_path, image_name)
            if os.path.isfile(image_path):
                os.rename(image_path, os.path.join(person_folder, image_name))

        # Update face encodings
        face_embeddings.generate_face_encodings()

        # Prepare data to add to Firebase
        date_str = datetime.now().strftime('%d-%m-%Y')
        ref = db.reference('Attendance system')
        member_data = {
            "name": name,
            "Role": role,
            "email": email,  # Add email here
            date_str: {  # Initialize today's attendance with empty entry and exit times
                "entry_time": "",
                "exit_time": ""
            }
        }

        # Add the new member's data to Firebase under the specified employee ID
        ref.child(employee_id).set(member_data)

        # Show success message with custom style
        self.show_custom_warning("Success", "Member added successfully", "info")

        # Navigate back to the manage members page
        self.manage_members_page()

    def remove_member(self, employee_id):
        # Remove from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        ref.delete()

        # Remove from local encodings folder
        encodings_folder_path = os.path.join('Encodings', employee_id)
        if os.path.exists(encodings_folder_path):
            # Remove all files in the encodings folder
            for file_name in os.listdir(encodings_folder_path):
                file_path = os.path.join(encodings_folder_path, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            # Now remove the folder itself
            os.rmdir(encodings_folder_path)

        # Remove images folder from Images directory
        images_folder_path = os.path.join('Images', employee_id)
        if os.path.exists(images_folder_path):
            shutil.rmtree(images_folder_path)

        self.show_custom_warning("Success", "Member removed successfully", "info")
        self.manage_members_page()

    def show_member_details_remove(self):
        employee_id = self.remove_id_entry.text()

        if not employee_id:
            self.show_custom_warning("Input Error", "Please enter an Employee ID.", "warning")
            return

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            self.show_custom_warning("Not Found", "No member found with the provided ID.", "warning")
            return

        # Clear the main layout for the new form
        self.clear_frame()

        # Frame for details
        frame = QFrame(self)
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setMinimumSize(320, 400)  # Increased height for better visibility of email details
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Allow resizing based on content

        # Main layout for the frame
        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Member Details", frame)
        title.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Centered form layout for labels and data
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(50, 20, 50, 20)

        # Employee ID, Name, and Email Display
        details_text = f"Employee ID: {employee_id}\nName: {member_details['name']}\nEmail: {member_details['email']}"
        member_details_label = QLabel(details_text, frame)
        member_details_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        member_details_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(member_details_label)

        layout.addLayout(form_layout)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Remove Button
        remove_button = QPushButton("Remove", frame)
        remove_button.setFixedSize(120, 40)
        remove_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(5, 161, 88, 0.8); 
                color: white; 
                border-radius: 10px;  
                font: bold 14px 'Segoe UI';  
                padding: 8px;  
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(9, 140, 78, 0.8); 
            }
            QPushButton:pressed {
                background-color: rgba(5, 109, 60, 0.8);
            }"""
        )
        remove_button.clicked.connect(lambda: self.remove_member(employee_id))
        button_layout.addWidget(remove_button)

        # Add button layout to the form layout and center it
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)

        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def show_member_details_view(self):
        employee_id = self.view_id_entry.text()

        if not employee_id:
            self.show_custom_warning("Input Error", "Please enter an Employee ID.", "warning")
            return

        # Fetch member details from Firebase
        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            self.show_custom_warning("Not Found", "No member found with the provided ID.", "warning")
            return

        # Clear the main layout for the new form
        self.clear_frame()

        # Frame for details
        frame = QFrame(self)
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(400, 380)  # Enlarged size for better display

        # Main layout for the frame
        layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Member Details", frame)
        title.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Centered form layout for labels and data
        form_layout = QVBoxLayout()
        form_layout.setContentsMargins(50, 20, 50, 20)

        # Employee ID, Name, and Email Display
        details_text = f"Employee ID: {employee_id}\nName: {member_details['name']}\nEmail: {member_details['email']}"
        member_details_label = QLabel(details_text, frame)
        member_details_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        member_details_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(member_details_label)

        layout.addLayout(form_layout)

        # Horizontal layout for the back button
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet(
            """QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }"""
        )
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Add button layout to the form layout and center it
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)

        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def view_and_edit_employee(self):
        employee_id = self.data_id_entry.text()
        if not employee_id:
            self.show_custom_warning("Input Error", "Please enter an Employee ID.", "warning")
            return

        ref = db.reference(f'Attendance system/{employee_id}')
        member_details = ref.get()

        if not member_details:
            self.show_custom_warning("Not Found", "No member found with the provided ID.", "warning")
            return

        # Clear the main layout for the new form
        self.clear_frame()

        # Frame for details
        frame = QFrame(self)
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(400, 500)  # Increased height for the email field
        self.main_layout.addWidget(frame)

        # Main layout for the frame
        layout = QVBoxLayout(frame)

        # Title
        title_label = QLabel("Edit Member Details", frame)
        title_label.setStyleSheet("color: white; font: bold 24px 'Segoe UI'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Centered form layout for labels and inputs
        form_layout = QGridLayout()
        form_layout.setContentsMargins(50, 20, 50, 20)  # Add margins for spacing

        # Employee ID Label and Value
        employee_id_label = QLabel("Employee ID:", frame)
        employee_id_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        employee_id_value = QLabel(employee_id, frame)
        employee_id_value.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        employee_id_value.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(employee_id_label, 0, 0, 1, 2, alignment=Qt.AlignCenter)
        form_layout.addWidget(employee_id_value, 1, 0, 1, 2, alignment=Qt.AlignCenter)

        # Spacer after Employee ID value
        form_layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Fixed), 2, 0, 1, 2)

        # Name Label and Entry
        name_label = QLabel("Name:", frame)
        name_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        self.name_entry = QLineEdit(frame)
        self.name_entry.setText(member_details.get('name', ''))
        self.name_entry.setFixedWidth(200)
        self.name_entry.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7); 
            color: #343A40; 
            font: 14px 'Segoe UI'; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
        """)
        form_layout.addWidget(name_label, 3, 0, 1, 2, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.name_entry, 4, 0, 1, 2, alignment=Qt.AlignCenter)

        # Role Label and Entry
        role_label = QLabel("Role:", frame)
        role_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        self.role_entry = QLineEdit(frame)
        self.role_entry.setText(member_details.get('Role', ''))
        self.role_entry.setFixedWidth(200)
        self.role_entry.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7); 
            color: #343A40; 
            font: 14px 'Segoe UI'; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
        """)
        form_layout.addWidget(role_label, 5, 0, 1, 2, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.role_entry, 6, 0, 1, 2, alignment=Qt.AlignCenter)

        # Email Label and Entry
        email_label = QLabel("Email:", frame)
        email_label.setStyleSheet("color: white; font: bold 18px 'Segoe UI'; background: transparent;")
        self.email_entry = QLineEdit(frame)
        self.email_entry.setText(member_details.get('email', ''))
        self.email_entry.setFixedWidth(200)
        self.email_entry.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.7); 
            color: #343A40; 
            font: 14px 'Segoe UI'; 
            padding: 10px; 
            border-radius: 15px; 
            box-shadow: 6px 6px 15px rgba(0, 0, 0, 0.2), 
                        -6px -6px 15px rgba(255, 255, 255, 0.9);
        """)
        form_layout.addWidget(email_label, 7, 0, 1, 2, alignment=Qt.AlignCenter)
        form_layout.addWidget(self.email_entry, 8, 0, 1, 2, alignment=Qt.AlignCenter)

        layout.addLayout(form_layout)

        # Horizontal layout for buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(20)

        # Update Button
        update_button = QPushButton("Update", frame)
        update_button.setFixedSize(120, 40)
        update_button.setStyleSheet("""
            QPushButton {
                background-color: rgba(5, 161, 88, 0.8); 
                color: white; 
                border-radius: 15px;  
                font: bold 14px 'Segoe UI';  
                padding: 8px;  
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(9, 140, 78, 0.8); 
            }
            QPushButton:pressed {
                background-color: rgba(5, 109, 60, 0.8);
            }
        """)
        update_button.clicked.connect(lambda: self.update_employee(employee_id))
        button_layout.addWidget(update_button)

        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(120, 40)
        back_button.setStyleSheet("background-color: #CD5C5C; color: white;")
        back_button.clicked.connect(self.manage_members_page)
        button_layout.addWidget(back_button)

        # Add button layout to the main layout
        layout.addLayout(button_layout)
        layout.setAlignment(button_layout, Qt.AlignCenter)
        # Add frame to main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)

    def update_employee(self, employee_id):
        name = self.name_entry.text()
        position = self.role_entry.text()
        email = self.email_entry.text()

        if not name or not position or not email:
            QMessageBox.warning(self, "Input Error", "Please fill out all fields.")
            return

        ref = db.reference(f'Attendance system/{employee_id}')
        ref.update({
            'name': name,
            'Role': position,
            'email': email
        })
        QMessageBox.information(self, "Success", "Member details updated successfully!")
        self.manage_members_page()

    def show_custom_warning(self, title, message, message_type):
        # Create the message box
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)

        # Set the type of message box and icon based on the provided message_type
        if message_type == "info":
            msg_box.setIcon(QMessageBox.Information)
        elif message_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
        elif message_type == "error":
            msg_box.setIcon(QMessageBox.Critical)
        else:
            raise ValueError("Invalid message_type. Use 'info', 'warning', or 'error'.")

        # Set custom font
        msg_box.setFont(QFont("Segoe UI", 14, QFont.Bold))

        # Apply a more elaborate stylesheet for a stylish popup
        msg_box.setStyleSheet("""
               QMessageBox {
                   background-color: qlineargradient(
                       x1: 0, y1: 0, x2: 1, y2: 1,
                       stop: 0 #e3f2fd, stop: 1 #bbdefb
                   );
                   border: 2px solid #90caf9;
                   border-radius: 10px;
                   color: #0d47a1;
                   padding: 10px;
               }
               QLabel {
                   color: #0d47a1;   /* Dark blue color for the message text */
                   font-size: 12pt;
                   font-weight: bold;
               }
               QPushButton {
                   background-color: #0d47a1;  /* Dark blue button color */
                   color: #ffffff;             /* White text color */
                   font-weight: bold;
                   padding: 8px;
                   border-radius: 6px;
                   min-width: 80px;
               }
               QPushButton:hover {
                   background-color: #1565c0; /* Slightly lighter blue for hover */
               }
               QPushButton:pressed {
                   background-color: #003c8f; /* Even darker blue when clicked */
               }
           """)

        # Display the message box
        msg_box.exec_()

    def download_data_page(self):
        # Clear the current frame
        self.clear_frame()
        self.setWindowState(Qt.WindowMaximized)
        self.setMinimumSize(800, 600)
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setAlignment(Qt.AlignCenter)
        # Frame setup
        frame = QFrame(self)
        frame.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            box-shadow: 8px 8px 20px rgba(0, 0, 0, 0.2), 
                        -8px -8px 20px rgba(255, 255, 255, 0.9);
        """)
        frame.setFixedSize(320, 320)
        self.main_layout.addWidget(frame)
        # Layout within the frame
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        # Title label
        title_label = QLabel("Download Attendance Data", frame)
        title_label.setStyleSheet("color: white; font: 20px 'Arial'; background: transparent;")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        # Button layout for vertical alignment
        button_layout = QVBoxLayout()
        # Download Daily Attendance Button
        daily_button = QPushButton("Daily", frame)
        daily_button.setFixedSize(200, 40)
        daily_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
            """
        )
        daily_button.clicked.connect(self.download_daily_attendance)
        button_layout.addWidget(daily_button)
        # Download Custom Attendance Button
        custom_button = QPushButton("Custom", frame)
        custom_button.setFixedSize(200, 40)
        custom_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(0, 85, 127, 0.8);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(0, 70, 106, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(0, 44, 66, 0.8);
            }
            """
        )
        custom_button.clicked.connect(self.open_custom_date_dialog)
        button_layout.addWidget(custom_button)
        # Back Button
        back_button = QPushButton("Back", frame)
        back_button.setFixedSize(200, 40)
        back_button.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(205, 92, 92, 0.9);
                color: white;
                border-radius: 10px;
                font: bold 16px 'Segoe UI';
                padding: 8px;
                box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.2), 
                            -4px -4px 10px rgba(255, 255, 255, 0.7);
            }
            QPushButton:hover {
                background-color: rgba(185, 72, 72, 0.9);
            }
            QPushButton:pressed {
                background-color: rgba(165, 52, 52, 0.9);
            }
            """
        )
        back_button.clicked.connect(self.admin_panel_page)  # You need to implement the `go_back` method
        button_layout.addWidget(back_button)
        # Center button layout and add it to the frame layout
        button_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(button_layout)
        # Add frame to the main layout
        self.root_layout.addWidget(frame, alignment=Qt.AlignCenter)
    def download_daily_attendance(self):
        print("Download daily attendance button clicked")  # Debugging line
        date_str = datetime.now().strftime('%d-%m-%Y')
        attendance_data = self.fetch_attendance_from_firebase(date_str)
        print(f"Fetched attendance data: {attendance_data}")
        if attendance_data:
            self.save_to_excel(attendance_data, f"Daily_Attendance_{date_str}.xlsx")
            self.show_success_message("Daily attendance data downloaded successfully!")
        else:
            QMessageBox.warning(self, "No Data", f"No attendance data found for {date_str}.")

    def open_custom_date_dialog(self):
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Date Range")
        dialog.setModal(True)

        # Set dialog styling
        dialog.setStyleSheet("""
            QDialog {
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        dialog.setMinimumWidth(400)
        # Layout for the dialog
        layout = QVBoxLayout(dialog)
        layout.setSpacing(20)

        # Title label
        title_label = QLabel("Select a Date Range", dialog)
        title_label.setStyleSheet("""
            color: #333333;
            font: bold 16px 'Segoe UI';
            background: transparent;
            margin-bottom: 20px;
            text-align: center;
        """)
        layout.addWidget(title_label)

        # From date label and picker
        start_date_edit = QDateEdit()
        start_date_edit.setCalendarPopup(True)
        start_date_edit.setDate(QDate.currentDate())
        start_date_edit.setStyleSheet("""
            QDateEdit {
                font: 14px 'Segoe UI';
                padding: 10px;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QDateEdit::drop-down {
                border-left: 2px solid #4CAF50;
            }
        """)
        layout.addWidget(QLabel("From:", dialog))
        layout.addWidget(start_date_edit)

        # To date label and picker
        end_date_edit = QDateEdit()
        end_date_edit.setCalendarPopup(True)
        end_date_edit.setDate(QDate.currentDate())
        end_date_edit.setStyleSheet("""
            QDateEdit {
                font: 14px 'Segoe UI';
                padding: 10px;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                background-color: rgba(255, 255, 255, 0.8);
            }
            QDateEdit::drop-down {
                border-left: 2px solid #4CAF50;
            }
        """)
        layout.addWidget(QLabel("To:", dialog))
        layout.addWidget(end_date_edit)
        # Buttons for dialog (OK & Cancel)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
                border: none;
            }
            QPushButton {
                background-color: rgba(76, 175, 80, 0.8);
                color: white;
                font: bold 14px 'Segoe UI';
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: rgba(56, 142, 60, 0.8);
            }
            QPushButton:pressed {
                background-color: rgba(46, 125, 50, 0.8);
            }
        """)
        buttons.accepted.connect(lambda: self.download_custom_attendance(start_date_edit.date(), end_date_edit.date()))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        # Execute dialog
        dialog.exec_()

    def download_custom_attendance(self, start_date, end_date):
        # Convert QDate to string format
        start_date_str = start_date.toString("dd-MM-yyyy")
        end_date_str = end_date.toString("dd-MM-yyyy")
        # Fetch attendance within the date range
        attendance_data = []
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.toString("dd-MM-yyyy")
            daily_data = self.fetch_attendance_from_firebase(date_str)
            if daily_data:
                attendance_data.extend(daily_data)
            current_date = current_date.addDays(1)
        if attendance_data:
            # Save data to Excel
            self.save_to_excel(attendance_data, f"Custom_Attendance_{start_date_str}_to_{end_date_str}.xlsx")
            self.show_success_message(
                f"Custom attendance data from {start_date_str} to {end_date_str} downloaded successfully!")
        else:
            QMessageBox.warning(self, "No Data",
                                f"No attendance data found between {start_date_str} and {end_date_str}.")

    def fetch_attendance_from_firebase(self, date_str):
        # Fetch attendance for the specific date from Firebase
        attendance_data = []
        attendance_ref = db.reference('Attendance system')
        try:
            all_data = attendance_ref.get()
            if all_data:
                for user_id, data in all_data.items():
                    # Check if the user has the specific date entry
                    if date_str in data:
                        # Retrieve the user's name and attendance data
                        name = data.get('name', 'N/A')  # Default to 'N/A' if name is not found
                        entry = data[date_str].get('entry_time', '')
                        exit = data[date_str].get('exit_time', '')

                        # Add the user's data to the attendance list
                        attendance_data.append({
                            'User ID': user_id,
                            'Name': name,
                            'Date': date_str,
                            'Entry Time': entry,
                            'Exit Time': exit
                        })
            else:
                print("No data found in Firebase.")
        except Exception as e:
            print(f"Error fetching data from Firebase: {e}")
        return attendance_data

    def save_to_excel(self, data, file_name):
        print(f"Saving to Excel: {data}")
        try:
            if data:  # Ensure data isn't empty before saving
                df = pd.DataFrame(data)
                df.to_excel(file_name, index=False)
                print(f"Attendance data saved to {file_name}")
            else:
                print("No data available to save")
        except Exception as e:
            print(f"Error saving to Excel: {e}")

    def show_success_message(self, message):
        # Create a customized QMessageBox
        success_message = QMessageBox(self)
        # Set the icon to a checkmark (success icon)
        success_message.setIcon(QMessageBox.Information)
        # Set the title of the message box
        success_message.setWindowTitle("Success!")
        # Set the message text
        success_message.setText(message)
        # Customize the message box style (background color, font, etc.)
        success_message.setStyleSheet("""
            QMessageBox {
                background-color: rgba(255, 255, 255, 1);
                font: 14px 'Segoe UI';
                color: white;
                border-radius: 10px;
                padding: 15px;
            }
            QMessageBox QPushButton {
                background-color: rgba(0, 120, 0, 0.8);
                color: white;
                font: bold 12px 'Segoe UI';
                border-radius: 5px;
                padding: 10px;
            }
            QMessageBox QPushButton:hover {
                background-color: rgba(0, 150, 0, 0.9);
            }
            QMessageBox QPushButton:pressed {
                background-color: rgba(0, 100, 0, 0.8);
            }
        """)
        # Add a button to close the message box (by default 'OK' is added)
        success_message.setStandardButtons(QMessageBox.Ok)
        # Set the alignment of the message
        success_message.setTextFormat(Qt.RichText)
        # Show the message box
        success_message.exec_()

    def update_entry_camera_source(self, index):
        if hasattr(self, 'cap1') and self.cap1.isOpened():
            self.cap1.release()  # Release the current camera
        self.entry_camera_source = index
        self.start_video_streams()  # Restart the video stream for Entry camera

    def update_exit_camera_source(self, index):
        if hasattr(self, 'cap2') and self.cap2.isOpened():
            self.cap2.release()  # Release the current camera
        self.exit_camera_source = index
        self.start_video_streams()  # Restart the video stream for Exit camera

    def start_video_streams(self):
        # Stop existing threads first
        self.stop_video()
        
        self.camera_running = True
        self.camera_thread_1 = threading.Thread(target=self.update_video_feed_1, daemon=True)
        self.camera_thread_2 = threading.Thread(target=self.update_video_feed_2, daemon=True)
        self.camera_thread_1.start()
        self.camera_thread_2.start()

    def update_video_feed_1(self):
        try:
            self.cap1 = cv2.VideoCapture(self.entry_camera_source)
            # Set MacBook camera resolution
            self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap1.set(cv2.CAP_PROP_FPS, 30)
            
            while self.camera_running:
                ret, frame = self.cap1.read()
                if not ret:
                    print("Failed to grab frame from Entry camera.")
                    break
                try:
                    processed_frame = self.recognize_face(frame, "Entry")
                    self.display_frame(processed_frame, self.entry_camera_feed)
                except Exception as e:
                    print(f"Error processing Entry camera frame: {e}")
                    self.display_frame(frame, self.entry_camera_feed)  # Show raw frame if processing fails
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Entry camera thread error: {e}")
        finally:
            if hasattr(self, 'cap1') and self.cap1 is not None:
                self.cap1.release()

    def update_video_feed_2(self):
        try:
            # For MacBook with single camera, create a mirrored feed or disable exit camera
            if self.exit_camera_source == self.entry_camera_source:
                # Use the same camera source but with different processing
                self.cap2 = cv2.VideoCapture(self.exit_camera_source)
                # Set MacBook camera resolution
                self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap2.set(cv2.CAP_PROP_FPS, 30)
            else:
                self.cap2 = cv2.VideoCapture(self.exit_camera_source)
                
            while self.camera_running:
                ret, frame = self.cap2.read()
                if not ret:
                    print("Failed to grab frame from Exit camera.")
                    # Show a placeholder image for exit camera if no second camera
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "Exit Camera", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(placeholder, "Not Available", (190, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    self.display_frame(placeholder, self.exit_camera_feed)
                    break
                try:
                    processed_frame = self.recognize_face(frame, "Exit")
                    self.display_frame(processed_frame, self.exit_camera_feed)
                except Exception as e:
                    print(f"Error processing Exit camera frame: {e}")
                    self.display_frame(frame, self.exit_camera_feed)  # Show raw frame if processing fails
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"Exit camera thread error: {e}")
        finally:
            if hasattr(self, 'cap2') and self.cap2 is not None:
                self.cap2.release()

    def display_frame(self, frame, label):
        try:
            # Check if label still exists
            if label is None or not hasattr(label, 'setPixmap'):
                return
                
            # Convert the frame to RGB for PyQt5
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = frame_rgb.shape
            bytes_per_line = channels * width
            q_image = QtGui.QImage(frame_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(q_image)
            # Display the image on the label
            label.setPixmap(pixmap)
        except Exception as e:
            print(f"Error displaying frame: {e}")

    def recognize_face(self, frame, camera_id):
        """
        Recognizes faces in a frame, compares the detected face with known encodings, and marks attendance.

        Args:
            frame (np.array): The current frame captured from the camera.
            camera_id (str): Identifier to differentiate between 'Entry' and 'Exit' cameras.

        Returns:
            np.array: The processed frame with annotations for recognized faces.
        """
        # Get today's date in the format used in Firebase
        today_date = datetime.now().strftime("%d-%m-%Y")

        # Convert the frame from BGR to RGB (as MTCNN expects RGB images)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, _ = self.mtcnn.detect(img)

        if boxes is not None:
            for box in boxes:
                try:
                    # Convert box coordinates to integers
                    box = box.astype(int).flatten()
                    x1, y1, x2, y2 = box

                    # Crop the face and resize it to 160x160 pixels
                    face = img[y1:y2, x1:x2]
                    
                    # Check if face crop is valid
                    if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
                        print(f"Invalid face crop dimensions: {face.shape}")
                        continue
                        
                    face_resized = cv2.resize(face, (160, 160))

                    # Preprocess the face and get the embedding
                    face_tensor = self.mtcnn(face_resized)
                    
                    if face_tensor is not None:
                        face_tensor = face_tensor.to(self.device)
                        face_tensor = face_tensor.squeeze(0)  # Remove the batch dimension
                        face_embedding = self.facenet_model(face_tensor.unsqueeze(0)).detach().cpu().numpy()

                        min_dist = float('inf')
                        identity = None

                        # Loop through each person in the known encodings and calculate the distance
                        for person, encodings in self.known_encodings.items():
                            # If we have multiple encodings for a person, check each one
                            for encoding in encodings:
                                dist = np.linalg.norm(face_embedding - encoding)
                                if dist < min_dist:
                                    min_dist = dist
                                    identity = person

                        # Set a threshold for recognition
                        threshold = 0.8

                        if min_dist < threshold:  # If the distance is less than the threshold, recognize the person
                            best_name = identity
                            label = f"{best_name}: {min_dist:.2f}"
                        else:
                            best_name, label = "Unknown", "Unknown"

                        # Draw a rectangle around the face and add the label (name) to the frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Mark attendance and send email if not already recorded/sent
                        if best_name != "Unknown":
                            current_time = datetime.now().strftime("%H:%M:%S")  # Capture the current time

                            # Fetch user data from the database
                            user_ref = db.reference(f'Attendance system/{best_name}')
                            user_data = user_ref.get()

                            if user_data:
                                # Check if attendance is already marked for today
                                attendance_ref = user_ref.child(today_date)
                                current_attendance = attendance_ref.get()

                                # Marking Entry
                                if camera_id == "Entry":
                                    # Only mark entry if not already marked
                                    if not current_attendance or not current_attendance.get('entry_time'):
                                        attendance_ref.update({
                                            'entry_time': current_time
                                        })
                                        self.mark_attendance(user_data.get('name', best_name), best_name, "entry")
                                        self.show_attendance_message(user_data.get('name', best_name))

                                        # Send email only if not sent before
                                        if not self.email_sent.get((best_name, today_date, "entry"), False):
                                            self.send_email(
                                                recipient_email=user_data.get('email', ''),
                                                name=user_data.get('name', best_name),
                                                mode="entry",
                                                time=current_time
                                            )
                                            # Mark email as sent
                                            self.email_sent[(best_name, today_date, "entry")] = True
                                    else:
                                        print(
                                            f"Entry attendance already marked for {best_name} on {today_date} at {current_attendance.get('entry_time')}")

                                # Marking Exit
                                elif camera_id == "Exit":
                                    # Only mark exit if entry is already marked and exit not marked
                                    if current_attendance and current_attendance.get(
                                            'entry_time') and not current_attendance.get('exit_time'):
                                        attendance_ref.update({
                                            'exit_time': current_time
                                        })
                                        self.mark_attendance(user_data.get('name', best_name), best_name, "exit")
                                        self.show_attendance_message(user_data.get('name', best_name))

                                        # Send email only if not sent before
                                        if not self.email_sent.get((best_name, today_date, "exit"), False):
                                            self.send_email(
                                                recipient_email=user_data.get('email', ''),
                                                name=user_data.get('name', best_name),
                                                mode="exit",
                                                time=current_time
                                            )
                                            # Mark email as sent
                                            self.email_sent[(best_name, today_date, "exit")] = True
                                    else:
                                        # Print different messages for different scenarios
                                        if not current_attendance:
                                            print(f"No entry attendance found for {best_name} on {today_date}")
                                        elif not current_attendance.get('entry_time'):
                                            print(f"No entry time recorded for {best_name} on {today_date}")
                                        elif current_attendance.get('exit_time'):
                                            print(
                                                f"Exit attendance already marked for {best_name} on {today_date} at {current_attendance.get('exit_time')}")

                except Exception as e:
                    print(f"An error occurred while processing the face from {camera_id}: {e}")
        else:
            print(f"No faces detected in {camera_id}.")

        return frame

    def send_email(self, recipient_email, name, mode, time):
        """
        Sends an email notification to the recognized person using yagmail.

        Args:
            recipient_email (str): Email address of the recipient.
            name (str): Name of the recognized person.
            mode (str): Attendance mode ('entry' or 'exit').
            time (str): Timestamp of the attendance.
        """
        sender_email = "srirambo1100@gmail.com"
        sender_password = "xmmi ohhj cvai pzws"  # Use app-specific password if 2FA is enabled

        try:
            # Initialize yagmail client
            yag = yagmail.SMTP(user=sender_email, password=sender_password)

            # Email subject and body
            subject = "Attendance Marked"
            body = f"""
              Dear {name},

              Your attendance has been successfully recorded.
              Mode: {mode.capitalize()}
              Time: {time}

              Have a great day!

              Regards,
              Attendance System
            """

            # Send the email
            yag.send(to=recipient_email, subject=subject, contents=body)

            print(f"Email successfully sent to {recipient_email}")

        except Exception as e:
            print(f"Failed to send email to {recipient_email}: {e}")

    def mark_attendance(self, name, user_id, mode):
        now = datetime.now()
        date_str, time_str = now.strftime('%d/%m/%Y'), now.strftime('%H:%M:%S')
        # Load or create the attendance file
        try:
            df = pd.read_csv(self.attendance_file)
        except FileNotFoundError:
            df = pd.DataFrame(columns=['Name', 'Date', 'Entry Time', 'Exit Time', 'User ID'])
        # Check if an entry exists for the date and user
        entry_exists = ((df['Name'] == name) & (df['Date'] == date_str)).any()
        if mode == "entry":
            if entry_exists:
                df.loc[(df['Name'] == name) & (df['Date'] == date_str), 'Entry Time'] = time_str
                print(f"Entry time updated for {name} at {time_str}")
            else:
                new_entry = pd.DataFrame(
                    {'Name': [name], 'Date': [date_str], 'Entry Time': [time_str], 'Exit Time': [""],
                     'User ID': [user_id]})
                df = pd.concat([df, new_entry], ignore_index=True)
                print(f"Entry time marked for {name} at {time_str}")
        elif mode == "exit":
            if entry_exists:
                df.loc[(df['Name'] == name) & (df['Date'] == date_str), 'Exit Time'] = time_str
                print(f"Exit time updated for {name} at {time_str}")
            else:
                print(f"Entry time missing for {name}. Cannot mark exit time.")
        # Save to CSV
        df.to_csv(self.attendance_file, index=False)
        # Update Firebase with date-wise attendance
        self.update_firebase_attendance(user_id, date_str, mode, time_str)

    def update_firebase_attendance(self, user_id, date_str, mode, time_str):
        # Set the reference to the specific date node for the user
        date_node = date_str.replace('/', '-')  # Format date as 'dd-mm-yyyy' for Firebase
        user_ref = db.reference(f'Attendance system/{user_id}/{date_node}')
        user_data = user_ref.get()
        # Prepare data to update in Firebase
        if mode == "entry":
            update_data = {'entry_time': time_str}
            if not user_data:
                # New entry for the day, initialize attendance fields
                update_data['exit_time'] = ""  # Set exit time to empty initially
        elif mode == "exit":
            update_data = {'exit_time': time_str}
        # Update Firebase
        user_ref.update(update_data)
        print(f"Firebase attendance updated for user ID: {user_id} on {date_node}")

    def show_attendance_message(self, name):
        # Check if the person has already been welcomed
        if name not in self.recognized_people:
            # Add the name to the set of recognized people
            self.recognized_people.add(name)

            # Update the label with the recognized name
            #self.attendance_label.config(text=f"Welcome: {name}")

            # Speak the welcome message
            welcome_message = f"Welcome, {name}"
            self.engine.say(welcome_message)
            self.engine.runAndWait()

    def init_engine(self):
        self.engine = pyttsx3.init()

        # Get available voices
        voices = self.engine.getProperty('voices')
        # Set a specific voice by index (e.g., 1 for a female voice)
        self.engine.setProperty('voice', voices[2].id)  # Change the index as needed
        # Optional: Set the rate and volume
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    def stop_video(self):
        # Stop video feed safely
        try:
            self.camera_running = False
            
            # Wait for threads to finish
            if hasattr(self, 'camera_thread_1') and self.camera_thread_1 and self.camera_thread_1.is_alive():
                self.camera_thread_1.join(timeout=2)
            if hasattr(self, 'camera_thread_2') and self.camera_thread_2 and self.camera_thread_2.is_alive():
                self.camera_thread_2.join(timeout=2)
                
            if hasattr(self, 'cap1') and self.cap1 is not None:
                self.cap1.release()
                self.cap1 = None
            if hasattr(self, 'cap2') and self.cap2 is not None:
                self.cap2.release()
                self.cap2 = None
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error stopping video: {e}")

    def clear_frame(self):
        for i in reversed(range(self.root_layout.count())):
            item = self.root_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        # Force an update to ensure layout refresh
        self.root_layout.update()

    def load_encodings(self, directory):
        """Load known face encodings from the specified directory."""
        known_encodings = {}

        # Loop through the directory containing subdirectories for each person
        for person_folder in os.listdir(directory):
            person_folder_path = os.path.join(directory, person_folder)

            if os.path.isdir(person_folder_path):  # Check if it's a folder
                person_encodings = []

                # Loop through each .npy file in the person's folder
                for encoding_file in os.listdir(person_folder_path):
                    if encoding_file.endswith('.npy'):  # Check for .npy file extension
                        encoding_path = os.path.join(person_folder_path, encoding_file)
                        encoding = np.load(encoding_path)  # Load the encoding
                        person_encodings.append(encoding)  # Add encoding to list

                if person_encodings:
                    # Store all encodings for the person in the dictionary, key = person's folder name
                    known_encodings[person_folder] = np.array(person_encodings)

        return known_encodings

    def close_app(self):
        # Stop video feed before closing
        self.stop_video()
        # Close the application cleanly
        QApplication.quit()

    def closeEvent(self, event):
        try:
            self.stop_video()  # Stop the video capture
            # Wait for threads to finish
            if hasattr(self, 'camera_thread_1') and self.camera_thread_1.is_alive():
                self.camera_thread_1.join(timeout=1)
            if hasattr(self, 'camera_thread_2') and self.camera_thread_2.is_alive():
                self.camera_thread_2.join(timeout=1)
        except Exception as e:
            print(f"Error during close: {e}")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttendanceApp()
    window.show()
    sys.exit(app.exec_())
