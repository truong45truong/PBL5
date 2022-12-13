
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import mediapipe as mp
import numpy as np
import tensorflow as tf
import datetime
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from .models import UploadEX, UploadGym, UploadYoga

feature="2"
option_feature="1"

class handle_pose:
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
    def make_landmark_timestep(results):
        c_lm = []
        for id, lm in enumerate(results.pose_landmarks.landmark):
            c_lm.append(lm.x)
            c_lm.append(lm.y)
            c_lm.append(lm.z)
            c_lm.append(lm.visibility)
        return c_lm
    def draw_result_detect_action_on_image(label,Time,img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 30)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 1
        lineType = 1
        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        cv2.putText(img,str(Time),
                    (1800,30),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        return image

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
class detect:
    def __init__(self,model):
        self.model=model
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils
        self.handle_mediapipe=handle_pose
    def detect_physical(self,image):
        results = self.pose.process(image)
        if results.pose_landmarks:
            lm_list = np.array(self.handle_mediapipe.make_landmark_timestep(results))
            action=np.argmax(self.model.predict(lm_list.reshape(-1,132,)))
            return action
        return False
def draw_result_detect_action_on_image(label,Time, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    thickness = 1
    lineType = 1
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img,str(int(Time)),
                (10,50),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
def draw_result_detect_yoga_on_image(label,Time,img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    thickness = 1
    lineType = 1
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    cv2.putText(img,str(int(Time)),
                (10,50),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img
def detect_movements(action,K,orderOfAction):
    if orderOfAction == action and K[action] == False:
        return True
    return False
def setK(lenK):
    K=[]
    for i in range(0,lenK):
        K.append(False)
    return K


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle
def draw_landmark_on_image(results,img):
    mpDraw = mp.solutions.drawing_utils
    mpPose = mp.solutions.pose
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    if(results.pose_landmarks):
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img
def draw_result_pushUp_on_image(reps,stage,angle,elbow,img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 0.5
    fontColor = (0, 255, 0)
    thickness = 1
    lineType = 1
    cv2.rectangle(img, (0,0), (225,73), (245,117,16), -1)
    cv2.putText(img, str(angle),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

    # Rep data
    cv2.putText(img, "REPS",
                    (15,12),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    cv2.putText(img, str(reps),
                    (10,60),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    # Stage data
    cv2.putText(img, 'STAGE',
                    (65,12),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    cv2.putText(img, str(stage),
                    (65,60),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
    return img
reps_count=0
config =tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.disable_eager_execution()
graph = tf.get_default_graph()
session = tf.compat.v1.Session(config=config)
with graph.as_default(), session.as_default():
    modelYoga = tf.keras.models.load_model("/home/pi/PBL/mysite/api/model/model_Yoga.hdf5",compile=False)
    modelEx1 = tf.keras.models.load_model("/home/pi/PBL/mysite/api/model/model_EX1.hdf5",compile=False)
    modelEx2 = tf.keras.models.load_model("/home/pi/PBL/mysite/api/model/model_EX2.hdf5",compile=False)
    modelEx3 = tf.keras.models.load_model("/home/pi/PBL/mysite/api/model/model_EX3.hdf5",compile=False)
    modelEx4 = tf.keras.models.load_model("/home/pi/PBL/mysite/api/model/model_EX4.hdf5",compile=False)
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
modelEx=[modelEx1,modelEx2,modelEx3,modelEx4]
def gen(camera,feature,option_feature):
    classes=['anantasana','bakasana','balasana','bhekasana']
    classesYoga=['1','2','3','4']
    K=setK(len(classes))
    stage = None
    modeldetect=detect(modelEx1)
    modeldetectyoga=detect(modelYoga)
    orderOfAction=0
    unit=1
    count=1
    while (True):
        count+=1
        img = camera.get_frame()
        results=pose.process(img)
        timecount=(count*unit)/11
        #sendmessage(str(int(timecount)),add[0])
        if(feature=="2"):
            # nhan dien dong tac
            with graph.as_default(),session:
                action=modeldetect.detect_physical(img)
            if detect_movements(action,K,orderOfAction)==True:
                img=draw_landmark_on_image(results=results,img=img)
                if(timecount==0):
                    #cout down
                    K[action]=True
                    orderOfAction=orderOfAction+1
                    count=0
                    unit=0
                img = draw_result_detect_action_on_image(label=classes[action],Time=timecount,img= img)
            else:
                note=str("dong tac dang thuc hien: "+classes[action] +" dong tac hien tai la: " + str(orderOfAction))
                img =draw_result_detect_action_on_image(label=note,Time=timecount,img=img)
                count=0
                unit=0
            img=draw_landmark_on_image(results=results,img=img)
        if(feature=="1"):
            if(results.pose_landmarks):
                landmarks =results.pose_landmarks.landmark
                # Get coordinates
                shoulder_left = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
                shoulder_right = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

                RIGHT_HIP = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
                RIGHT_KNEE = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
                RIGHT_ANKLE = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angle
                if(option_feature=="dumbbellcurl"):
                    angle = calculate_angle(shoulder_left, elbow, wrist)
                if(option_feature=="squats"):
                    angle = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
                if(option_feature=="crunch"):
                    angle = calculate_angle(shoulder_right, RIGHT_HIP, RIGHT_KNEE)
                # Visualize angle
                if(option_feature!=""):
                    if angle > 160:
                            stage = "down"
                    if angle < 30 and stage =='down':
                            stage="up"
                            counter +=1
                if(option_feature=="dumbbellcurl"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,elbow,img)
                if(option_feature=="squats"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_KNEE,img)
                if(option_feature=="crunch"):
                    img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_HIP,img)

                # Render detections
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                        mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                        mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                        )
        if(feature=="3"):
            if(results.pose_landmarks):
                yogaAction=option_feature
                #with graph.as_default(),session.as_default():
                    #action=modeldetectyoga.detect_physical(img)
                #img=draw_landmark_on_image(results=results,img=img)
                action=0
                if(action==yogaAction):
                    pass
                    #img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=timecount,img= img)
                else:
                    count=0
                    unit=0
                    #img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=timecount,img= img)
        _, jpeg = cv2.imencode('.jpg', img)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
def genYoga(camera,yogaoption):
    modelyoga = tf.keras.models.load_model("C:/Users/DELL/OneDrive/Máy tính/Django/PBL/mysite/api/model/model_Yoga.h5")
    modeldetectyoga=detect(modelyoga)
    count=1
    unit=1
    classesYoga=[]
    while (True):
        count+=1
        img = camera.get_frame()
        results=pose.process(img)
        timecount=(count*unit)/11
        if(results.pose_landmarks):
            action=modeldetectyoga.detect_physical(img)
            img=draw_landmark_on_image(results=results,img=img)
            if(action==yogaoption):
                img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=timecount,img= img)
            else:
                count=0
                unit=0
                img = draw_result_detect_yoga_on_image(label=classesYoga[action],Time=timecount,img= img)
        _, jpeg = cv2.imencode('.jpg', img)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
def genGym(camera,gymoption):
    while True:
        img = camera.get_frame()
        results=pose.process(img)
        if(results.pose_landmarks):
            landmarks =results.pose_landmarks.landmark
            # Get coordinates
            shoulder_left = [landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulder_right = [landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mpPose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mpPose.PoseLandmark.LEFT_WRIST.value].y]

            RIGHT_HIP = [landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mpPose.PoseLandmark.RIGHT_HIP.value].y]
            RIGHT_KNEE = [landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_KNEE.value].y]
            RIGHT_ANKLE = [landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mpPose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Calculate angle
            if(gymoption=="dumbbellcurl"):
                angle = calculate_angle(shoulder_left, elbow, wrist)
            if(gymoption=="squats"):
                angle = calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
            if(gymoption=="crunch"):
                angle = calculate_angle(shoulder_right, RIGHT_HIP, RIGHT_KNEE)
            # Visualize angle
            if(gymoption!=""):
                if angle > 160:
                        stage = "down"
                if angle < 30 and stage =='down':
                        stage="up"
                        counter +=1
            if(gymoption=="dumbbellcurl"):
                img= draw_result_pushUp_on_image(counter,stage,angle,elbow,img)
            if(gymoption=="squats"):
                img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_KNEE,img)
            if(gymoption=="crunch"):
                img= draw_result_pushUp_on_image(counter,stage,angle,RIGHT_HIP,img)

            # Render detections
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS,
                                    mpDraw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                    mpDraw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
def genPhysical(camera,option):
     while (True):
        count+=1
        img = camera.get_frame()
        results=pose.process(img)
        timecount=(count*unit)/11
        k=0
        classes=["dong tac 1","dong tac 2","dong tac 3","dong tac4"]
        if(feature=="2"):
            # nhan dien dong tac
            with graph.as_default(),session:
                action=modelEx[option].detect_physical(img)
            if K:
                img=draw_landmark_on_image(results=results,img=img)
                if(timecount==3):
                    #cout down
                    K[action]=True
                    orderOfAction=orderOfAction+1
                    count=0
                    unit=0

                img = draw_result_detect_action_on_image(label=classes[action],Time=timecount,img= img)
            else:
                note=str("dong tac dang thuc hien: "+classes[action] +" dong tac hien tai la: " + str(orderOfAction))
                img =draw_result_detect_action_on_image(label=note,Time=timecount,img=img)
                count=0
                unit=0
            img=draw_landmark_on_image(results=results,img=img)

cam=VideoCamera()
def Add_El():
    Date=datetime.date.today()
    Gym=UploadGym(Date,"0","0","0")
    Yoga=UploadYoga(Date,"0")
    Ex=UploadEX(Date,"0")
    if Gym is not None:
            Gym.save()
            Yoga.save()
            Ex.save()

def Upload_GYM(Date,squat,crunch,dumbbellcurl):
    #b.update(squat=squat, crunch=crunch, Yoga=dumbbellcurl)
    UploadGym.objects.filter(date=Date).update(squat=squat, crunch=crunch, yoga=dumbbellcurl)
    
def runUpload():
    while True:
        print(reps_count)
        Hour=datetime.datetime.now().hour
        if(Hour==0):
            Add_El()

@gzip.gzip_page
def livefe_Gym_squats(request):
    try:
        return StreamingHttpResponse(genGym(cam,"squats"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Gym_crunch(request):
    try:
        return StreamingHttpResponse(genGym(cam,"crunch"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Gym_dumbbellcurl(request):
    try:
        return StreamingHttpResponse(genGym(cam,"dumbbellcurl"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Ex_lession1(request):
    try:
        return StreamingHttpResponse(genPhysical(cam,"0"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Ex_lession2(request):
    try:
        return StreamingHttpResponse(genPhysical(cam,"1"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Ex_lession3(request):
    try:
        return StreamingHttpResponse(genPhysical(cam,"2"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Ex_lession4(request):
    try:
        return StreamingHttpResponse(genPhysical(cam,"3"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Yoga_lession1(request):
    try:
        return StreamingHttpResponse(genYoga(cam,"0"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Yoga_lession2(request):
    try:
        return StreamingHttpResponse(genYoga(cam,"1"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Yoga_lession3(request):
    try:
        return StreamingHttpResponse(genYoga(cam,"2"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass
def livefe_Yoga_lession4(request):
    try:
        return StreamingHttpResponse(genYoga(cam,"3"), content_type="multipart/x-mixed-replace;boundary=frame")    
    except:  # This is bad! replace it with proper handling
        pass