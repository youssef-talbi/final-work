import logging
import cv2
import json
import face_recognition
import numpy as np
from flask import Blueprint, jsonify, request,render_template
from flask import Flask, session
from flask_session import Session
import sys
import face_recognition
from flask_login import LoginManager, login_user, current_user, logout_user, login_required
import numpy as np
sys.path.append('C:\\Users\\youssef\\Desktop\\PFE\\demo2\\database\\Models')
sys.path.append('C:\\Users\\youssef\\Desktop\\PFE\\demo2')
sys.path.append('C:\\Users\\youssef\\Desktop\\PFE\\demo2\\utils')
sys.path.append('C:\\Users\\youssef\\Desktop\\PFE\\demo2\\Routes')
from flask import  Response,  send_from_directory
from flask import Flask, render_template
from flask_login import LoginManager
from flask_socketio import SocketIO, send, emit, join_room, leave_room
from flask_cors import CORS
from database import db
from employee_routes import *
from chat_route import chat_bp
from departments_routes import *
from roles_routes import *
from applicant_routes import *
from benefitprograms import *
from auth import *
from traininprog_routes import *
from leaverequest_routes import *
from attendance_routes import *
from audit_train_routes import *
from manager_review import *
from count_employee import *
from upload_images import *
from benefitselection import *
from training_selection import *
from Integrated_Calendar import *
from job_routes import *
from performance_review import *
from performance_goals import *
from performance_reports import *
from Employee import Employee
from Chat import Chat

app = Flask(__name__)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

# Configure MySQL database URI
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/systemhr'  # Replace root with your MySQL username, and add the password if required
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Enable CORS
CORS(app)
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return Employee.query.get(int(user_id))

app.secret_key='mysecretkey'

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


app.register_blueprint(employee_bp)
app.register_blueprint(departments_routes)
app.register_blueprint(roles_bp)
app.register_blueprint(benefits_bp)
app.register_blueprint(auth)
app.register_blueprint(traininprog_routes)
app.register_blueprint(leaverequest_routes)
app.register_blueprint(attendance_bp)
app.register_blueprint(leave_requests)
app.register_blueprint(chat_bp)
app.register_blueprint(count_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(selection_bp)
app.register_blueprint(blueprint=Tselection_bp)
app.register_blueprint(calendar_bp)
app.register_blueprint(job_bp)

@login_manager.user_loader
def load_user(user_id):
    return Employee.query.get(int(user_id))

@app.route('/')
def index():
    return render_template('login.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected:', request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@socketio.on('joinChannel')
def on_join(data):
    channel_id = data['channel_id']
    join_room(channel_id)
    print(f'Client joined channel {channel_id}')

@socketio.on('leaveChannel')
def on_leave(data):
    channel_id = data['channel_id']
    leave_room(channel_id)
    print(f'Client left channel {channel_id}')

@socketio.on('chatmsg')
def handle_message(data):
    channel_id = data['channel_id']
    message = data['msg']
    employee_id = session['employee_id']
    employee = Employee.query.get(employee_id)
    new_message = Chat(message=message, sender_id=employee_id, channel_id=channel_id)
    db.session.add(new_message)
    db.session.commit()
    
    emit('message_received', {
        'channel_id': channel_id,
        'message': message,
        'sender': f'{employee.FirstName} {employee.LastName}',
        'timestamp': new_message.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    }, room=channel_id)
    

def initialize_camera():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    return video_capture

def load_face_encodings():
    employees = Employee.query.all()
    face_encodings = {}
    for employee in employees:
        if employee.face_encoding:
            encodings_list = json.loads(employee.face_encoding)
            encodings = [np.array(encoding) for encoding in encodings_list]
            face_encodings[employee.EmployeeID] = {
                "name": f"{employee.FirstName} {employee.LastName}",
                "encodings": encodings
            }
    return face_encodings

recognized_today = []
today_str = datetime.now().strftime("%Y%m%d")
def create_attendance_record(employee_id):
    try:
        new_attendance = Attendance(
            AttendanceID=datetime.now().strftime("%Y%m%d") + str(employee_id),
            EmployeeID=employee_id,
            ClockInTime=datetime.now(),
            ClockOutTime=datetime.now()
        )
        db.session.add(new_attendance)
        db.session.commit()
        logging.info(f"Attendance record created for employee {employee_id}")
        return new_attendance
    except Exception as e:
        logging.error(f"Error creating attendance record for employee {employee_id}: {e}")
        db.session.rollback()
        return None

def process_frame(frame, face_encodings_db):
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    recognized_faces = []

    for face_encoding in face_encodings:
        matches = []
        for employee_id, employee_data in face_encodings_db.items():
            match = face_recognition.compare_faces(employee_data["encodings"], face_encoding, tolerance=0.6)
            if any(match):
                matches.append((employee_id, employee_data["name"]))
        
        if matches:
            best_match = matches[0]
            recognized_faces.append(best_match[1])
            if best_match[0] not in recognized_today:
                recognized_today.append(best_match[0])
                logging.info(f"{datetime.now()} -- first time seeing {best_match[0]}, their name is {best_match[1]}")
                create_attendance_record(best_match[0])
            else:
                attendance_id = today_str + str(best_match[0])
                attendance_record = Attendance.query.get(attendance_id)
                
                if attendance_record is None:
                    logging.error(f"Attendance record not found for ID: {attendance_id}")
                    continue
                
                second_appearance = datetime.now()
                clock_in = attendance_record.ClockInTime
                time_difference = second_appearance - clock_in
                clock_out = attendance_record.ClockOutTime
                if time_difference > timedelta(seconds=10) and clock_in == clock_out:
                    attendance_record.ClockOutTime = second_appearance
                    db.session.commit()

                logging.info(f"{datetime.now()} -- already seen {best_match[0]}, their name is {best_match[1]}, lapsed time is {time_difference}")
        else:
            recognized_faces.append("Unknown person")
            logging.info('unknown')
    
    return face_locations, recognized_faces

def generate_frames(face_encodings_db):
    video_capture = initialize_camera()
    with app.app_context():
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            face_locations, recognized_faces = process_frame(frame, face_encodings_db)
 
            for (top, right, bottom, left), name in zip(face_locations, recognized_faces):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        video_capture.release()

@app.route('/video_feed')
def video_feed():
    face_encodings_db = load_face_encodings()
    return Response(generate_frames(face_encodings_db), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True)