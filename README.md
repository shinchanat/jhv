# krtana a computer Vision package v0.0.1:-
### I have not yet upload to pypi.
## methods inside krtana's Detect class :-

* build_hands().
* build_face_mesh().
* build_pose().

# Installing process:-

* Make sure you have python3 installed in your system.
##### Run the following command in cmd to install krtana.

    
    pip install krtana


# Example:-

## Hand Tracking using krtana.
    import krtana

    webcam = krtana.VideoCapture(0)

    krtanaa = krtana.Detect()   

    while True:
	
        scc,img = webcam.read()
        img = krtana.flip(img,1)
        pos = krtanaa.build_hands(img)
        if pos:
             print(pos)
        krtana.imshow('screen_name',img)
        if krtana.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()         
    krtana.destroyAllWindows()

## Hand Tracking without using krtana.
    import cv2 as cv  
    import mediapipe as mp
    
    mphands = mp.solutions.hands  
    Hands = mphands.Hands()  
    mpdraw = mp.solutions.drawing_utils
    
    webcam = cv.VideoCapture(0)
    
    while True:
            
        scc,img = webcam.read()
        img = cv.flip(img,1)
        rgbimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results = Hands.process(img)
        hand_pos = []
        if results.multi_hand_landmarks:
            
            for hand_lm in results.multi_hand_landmarks:
                mpdraw.draw_landmarks(img,hand_lm,mphands.HAND_CONNECTIONS)
                for finger_id ,finger_pos in enumerate(hand_lm.landmark):
                    h,w,c = img.shape
                    x,y = int(finger_pos.x*w),int(finger_pos.y*h)
                    hand_pos.append([finger_id,x,y])
            print(hand_pos)
        cv.imshow('screen_name',img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()  
    cv.destroyAllWindows()

# Face Mesh using krtana:-

    import krtana

    webcam = krtana.VideoCapture(0)

    krtanaa = krtana.Detect()   

    while True:
	
        scc,img = webcam.read()
        img = krtana.flip(img,1)
        pos = krtanaa.build_face_mesh(img)
        if pos:
             print(pos)
        krtana.imshow('screen_name',img)
        if krtana.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()         
    krtana.destroyAllWindows()


# Face Mesh without using krtana:-

    import cv2 as cv
    import mediapipe as mp
    
    
    mpfacemesh = mp.solutions.face_mesh
    FaceMesh = mpfacemesh.FaceMesh()
    mpdraw = mp.solutions.drawing_utils
    drawspec = mpdraw.DrawingSpec(circle_radius = 0, thickness = 1)

    webcam = cv.VideoCapture(0)
    
    while True:
            
        scc,img = webcam.read()
        img = cv.flip(img,1)
        rgbimg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results = FaceMesh.process(img)
        face_pose = []
        if results.multi_face_landmarks:
            
            for face_lm in results.multi_face_landmarks:
                
                mpdraw.draw_landmarks(img,face_lm,
                                        mpfacemesh.FACEMESH_TESSELATION,
                                        drawspec,drawspec)
                for face_id ,face_pos in enumerate(face_lm.landmark):
                    h,w,c = img.shape
                    x,y = int(face_pos.x*w),int(face_pos.y*h)
                    
                    face_pose.append([face_id,x,y])
                
            print(face_pose)
        cv.imshow('screen_name',img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    webcam.release()  
    cv.destroyAllWindows()

# Pose using krtana:-
    
    import krtana
    
    webcam = krtana.VideoCapture(0)

    krtanaa = krtana.Detect()   

    while True:
	
        scc,img = webcam.read()
        img = krtana.flip(img,1)
        pos = krtanaa.build_pose(img)
        if pos:
             print(pos)
        krtana.imshow('screen_name',img)
        if krtana.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()         
    krtana.destroyAllWindows()

# Pose without using krtana:-

    import cv2 as cv
    import mediapipe as mp

    webcam = cv.VideoCapture(0)

    mppose = mp.solutions.pose
    Pose = mppose.Pose()
    mpdraw = mp.solutions.drawing_utils

    while True:
        scc,img = webcam.read()
        img = cv.flip(img,1)
        imgg = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        result = Pose.process(imgg)
        pose_position = []
        
        if result.pose_landmarks:
                mpdraw.draw_landmarks(img,result.pose_landmarks,mppose.POSE_CONNECTIONS)
                for ids,pos in enumerate(result.pose_landmarks.landmark):
                        h,w,c = img.shape
                        x,y = int(pos.x*w),int(pos.y*h)
                        pose_position.append([ids,x,y])
                print(pose_position[10])
        cv.imshow('screeen_name',img)
        if cv.waitKey(15) & 0xFF == ord('q'):
                break

    webcam.release()
    cv.destroyAllWindows()

# About krtana:-
* krtana  depends on google's mediapipe and opencv, it was created to apply 
machine learning in computer vision projects by just using few lines of code.
* since krtana depends on cv2 you can access all the functions and attributes that are available in cv2 version '4.5.3'.
