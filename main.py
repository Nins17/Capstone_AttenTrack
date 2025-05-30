from flask import Flask, render_template, request, redirect, url_for, session, flash
from flaskext.mysql import MySQL
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime, date
import pandas as pd
import joblib
import cv2
import face_recognition


app = Flask(__name__)
app.secret_key = "ams"

# MySQL Config
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'ams'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

# Initialize MySQL
mysql = MySQL()
mysql.init_app(app)

#global functions
def get_db_cursor():
    conn = mysql.connect()
    cursor = conn.cursor()
    return cursor, conn
def not_logged():
    flash('You are not logged in, Session failed!', 'error')
    return redirect(url_for('index'))

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")
nimgs = 10
imgBackground=cv2.imread("background.png")
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')
        

def totalreg():
    return len(os.listdir('static/faces'))

def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []

def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l

#routes
#landing
@app.route('/')
def index():
    return render_template("index.html")

#teacher
#<----pages---->
@app.route('/login_teacher_form', methods=["POST", "GET"])
def login_teacher_form():
    return render_template("user/login_teacher.html")

@app.route('/user_home', methods=["POST", "GET"])
def user_home():
    if session.get('logged_in')==True:
        logged_teacher= session.get('logged_teacher')
        return render_template("user/index.html",logged_teacher=logged_teacher)
    else:
        return not_logged()
 

@app.route('/enrollStudentform',methods=["POST", "GET"])
def sampleform():
    if session.get('logged_in')==True:
        logged_teacher= session.get('logged_teacher')
        cursor, conn = get_db_cursor()
        cursor.execute('SELECT DISTINCT grade_level FROM class_schedules WHERE teacher_id=%s',(session.get('teacher_id')))
        avail_grade_level = [row[0] for row in cursor.fetchall()]
        print(avail_grade_level)

        cursor.execute('SELECT DISTINCT section FROM class_schedules WHERE teacher_id=%s',(session.get('teacher_id')))
        avail_section = [row[0] for row in cursor.fetchall()]
        print(avail_section)
        cursor.execute('SELECT DISTINCT subject FROM class_schedules WHERE teacher_id=%s',(session.get('teacher_id')))
        avail_subject = [row[0] for row in cursor.fetchall()]
        print(avail_subject)
        
        if avail_grade_level or avail_section or avail_subject :
            return render_template("user/forms/enroll_student_form.html",avail_grade_level=avail_grade_level,avail_section=avail_section,avail_subject=avail_subject,logged_teacher=logged_teacher)
        else:
            pass
    else:
        return not_logged()
 
    
   
@app.route('/class_schedules',methods=["POST", "GET"])
def class_schedules():
    if session.get('logged_in')==True:
        logged_teacher= session.get('logged_teacher')
        cursor, conn = get_db_cursor()
        cursor.execute('SELECT * FROM class_schedules WHERE teacher_id=%s',(session.get('teacher_id')))
        class_scheds=cursor.fetchall()
        teacher_name=session.get('teacher_name')
        
        return render_template("user/class_schedules.html",class_scheds=class_scheds,teacher_name=teacher_name, logged_teacher= logged_teacher)
    else:
        return not_logged()
 
   
#<----actions---->
@app.route('/login_teacher', methods=["POST","GET"])
def login_teacher():
    if request.method == "POST":
        teacher_id =int(request.form["teacher_id"])
        teacher_pass =str(request.form["teacher_pass"])
        
        cursor,conn = get_db_cursor()
        cursor.execute('SELECT * FROM teacher_account WHERE teacher_ID=%s and teacher_password = %s',(teacher_id,teacher_pass))
        valid_teacher=cursor.fetchone()
        if valid_teacher:
            session["teacher_id"]=teacher_id
            session["teacher_pass"]=teacher_pass
            session["teacher_name"]=valid_teacher[1]
            session["logged_in"]=True
            cursor.execute('SELECT * FROM teacher_account WHERE teacher_ID=%s',(session.get('teacher_id')))
            logged_teacher=cursor.fetchall()
            if session.get('logged_in')==True:
                session["logged_teacher"]=logged_teacher
                return render_template("user/index.html",logged_teacher=logged_teacher)
            else:
                flash('Log In Failed', 'error')
                return redirect(url_for('login_teacher_form'))
        else:
            flash('No Account Found', 'error')
            return redirect(url_for('login_teacher_form'))
    else:
        flash('You are not logged in, Session failed!', 'error')
        return redirect(url_for('index'))

@app.route('/add_schedule',methods=["POST", "GET"])
def add_schedule():
    cursor, conn = get_db_cursor()
    if request.method == "POST":
        classGradeLevel = str(request.form["classGradeLevel"])
        class_section = str(request.form["class_section"])
        class_subject = str(request.form["class_subject"])
        class_schedule = str(request.form["class_schedule"])
        class_start_time = str(request.form["class_start_time"])
        class_end_time = str(request.form["class_end_time"])
        number_of_students=0
        teacher=str(session.get('teacher_name'))
        teacher_id=int(session.get('teacher_id'))
            
        query = """
            INSERT INTO class_schedules (
                `grade_level`, 
                `section`, 
                `subject`, 
                `schedule`, 
                `start_time`, 
                `end_time`, 
                `number_of_students`, 
                `teacher`,
                `teacher_id`
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values=(classGradeLevel,class_section,class_subject,class_schedule,class_start_time,class_end_time,number_of_students,teacher,teacher_id)    
        cursor.execute(query,values)
        conn.commit()
        return redirect('/class_schedules')

# @app.route('/enroll_student',methods=["POST", "GET"])
# def enroll_student():
#       if request.method == "POST":
#         student_id = request.form['student_id']
#         student_first_name = request.form['student_first_name']
#         student_middle_name = request.form['student_middle_name']
#         student_last_name = request.form['student_last_name']
#         student_suffix = request.form['student_suffix']
#         student_age = request.form['student_age']
#         student_gl = request.form['student_gl']
#         student_section = request.form['student_section']
#         student_subject = request.form['student_subject']
#         student_guardian = request.form['student_guardian']
#         guardian_contact = request.form['guardian_contact']
        
      
#         return redirect(url_for('index'))

@app.route("/take_attendance", methods=["POST","GET"])
def take_attendance():
    
    names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
            cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
            cv2.putText(frame, f'{identified_person}', (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)



#admin
@app.route('/admin_home', methods=["POST", "GET"])
def admin_home():
    return render_template("admin/index.html")

@app.route('/sign_out')
def sign_out():
    session.clear()
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
    