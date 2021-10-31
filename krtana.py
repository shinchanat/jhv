"""MIT LICENSE

Copyright 2021 Harish



Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation 
files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, 
modify, merge, publish,distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:



The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__all__ = ['bulid_hands','build_face_mesh','build_pose']


import mediapipe as mp
from cv2 import*

class Detect:

    def __init__(self,
               static_image_mode=False,
               max_num_hands=2,
               max_num_faces = 1,
               refine_landmarks=False,
               model_complexity=1,
               model_selection = 0,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):
        
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.max_num_faces = max_num_faces
        self.model_selection = model_selection
        self.refine_landmarks = refine_landmarks
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        
        self.mphands = mp.solutions.hands
        self.mppose = mp.solutions.pose
        self.mpfacemesh = mp.solutions.face_mesh
        self.mpfaces = mp.solutions.face_detection
        
        self.mpdraw = mp.solutions.drawing_utils

        
        self.Hands = self.mphands.Hands(self.static_image_mode,
                                        self.max_num_hands,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        
        self.FaceMesh = self.mpfacemesh.FaceMesh(self.static_image_mode,
                                        self.max_num_faces,
                                        self.refine_landmarks,
                                        self.min_detection_confidence,
                                        self.min_tracking_confidence)
        self.Pose = self.mppose.Pose()

        self.Face = self.mpfaces.FaceDetection()
        

    def build_hands(self,img,
                    hands_position = True,
                    hands_connection = True):
        
        hand_pos = []
        
        rgbimg = cvtColor(img,COLOR_BGR2RGB)
        hand_detect = self.Hands.process(rgbimg)
        
        if hand_detect.multi_hand_landmarks:
            for hand_landmark in hand_detect.multi_hand_landmarks:
                
                for ids,pos in enumerate(hand_landmark.landmark):
                    h,w,c = img.shape
                    x,y = int(pos.x*w),int(pos.y*h)

                    if hands_position == True:
                        hand_pos.append([ids,x,y])
                if hands_connection == True:
                    self.mpdraw.draw_landmarks(img,hand_landmark,self.mphands.HAND_CONNECTIONS)

        return hand_pos

    def build_face_mesh(self,img,
                        face_mesh_position = True,
                        face_mesh_connection = True,
                        thickness = 1,
                        radius = 0):
        
        face_pos = []
        mpdrawspec = self.mpdraw.DrawingSpec(thickness= thickness,circle_radius = radius)
        
        rgbimg = cvtColor(img,COLOR_BGR2RGB)
        face_detect = self.FaceMesh.process(rgbimg)
        
        if face_detect.multi_face_landmarks:
            for face_landmark in face_detect.multi_face_landmarks:
                
                for ids,pos in enumerate(face_landmark.landmark):
                    h,w,c = img.shape
                    x,y = int(pos.x*w),int(pos.y*h)
                    
                    if face_mesh_position == True:
                        face_pos.append([ids,x,y])
                        
                if face_mesh_connection == True:
                    self.mpdraw.draw_landmarks(img,face_landmark,
                                               self.mpfacemesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec = mpdrawspec,
                                               connection_drawing_spec = mpdrawspec)
                    
        return face_pos

    def build_pose(self,img,
                   pose_position = True,
                   pose_connection = True):

        body_pos = []
        
        rgbimg = cvtColor(img,COLOR_BGR2RGB)
        pos_detect = self.Pose.process(rgbimg)
        
        if pos_detect.pose_landmarks:

            if pose_connection == True:
                self.mpdraw.draw_landmarks(img,pos_detect.pose_landmarks,
                                           self.mppose.POSE_CONNECTIONS)
                
                for ids,pos in enumerate(pos_detect.pose_landmarks.landmark):
                    h,w,c = img.shape
                    x,y = int(pos.x*w),int(pos.y*h)

                    if pose_position == True:
                        body_pos.append([ids,x,y])
            else:
                
                for ids,pos in enumerate(pos_detect.pose_landmarks.landmark):
                    h,w,c = img.shape
                    x,y = int(pos.x*w),int(pos.y*h)

                    if pose_position == True:
                        body_pos.append([ids,x,y])

        return body_pos
