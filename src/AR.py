import argparse

import cv2
import numpy as np
import math
import os
from objloader_simple import *

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 70 # tạo biến chứa ngưỡng thấp nhất về độ tương đồng của ảnh input đầu vào và ảnh detect để render object.
DEFAULT_COLOR = (143, 141, 136) # tạo 1 màu mặc định cho các object không có nhiều màu.


def main():
    """
    This functions loads the target surface image,
    """
    homography = None 
    # matrix of camera parameters (made up but works quite well for me) 
    camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
    # create SIFT keypoint detecto
    orb = cv2.SIFT_create()
    # create Flann from OpenCV instead of BFMatcher to get a higher speed running time.
    # Flann requires two dictionaries(index_params and search_params) and its parameters are refered from opencv-website.
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    # load the reference surface that will be searched in the video stream
    dir_name = os.getcwd()
    model1 = cv2.imread(os.path.join(dir_name, 'D:\\augmented-reality-master\\reference\\book1.jpg'),0 )
    model2 = cv2.imread(os.path.join(dir_name, 'D:\\augmented-reality-master\\reference\\book2.jpg'),0 )
    model3 = cv2.imread(os.path.join(dir_name, 'D:\\augmented-reality-master\\reference\\book3.jpg'),0 )
    model = [model1,model2,model3]
    # Compute model keypoints and its descriptors
    kp1_model, des1_model = orb.detectAndCompute(model1, None)
    kp2_model, des2_model = orb.detectAndCompute(model2, None)
    kp3_model, des3_model = orb.detectAndCompute(model3, None)
    kp_model = [kp1_model, kp2_model, kp3_model]
    des_model = [des1_model, des2_model, des3_model]

    # Load 3D model from OBJ file
    obj1 = OBJ(os.path.join(dir_name, 'D:\\augmented-reality-master\\models\\pirate-ship-fat.obj'), swapyz=True)  
    obj2 = OBJ(os.path.join(dir_name, 'D:\\augmented-reality-master\\models\\motorbike.obj'), swapyz=True) 
    obj3 = OBJ(os.path.join(dir_name, 'D:\\augmented-reality-master\\models\\bus.obj'), swapyz=True)  
    obj = [obj1, obj2, obj3]
    # init video capture
    url = 'http://192.168.137.4:8080/video'
    cap = cv2.VideoCapture(0)
    while True:
        # read the current frame
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture video")
            return 
        # find and draw the keypoints of the frame
        kp_frame, des_frame = orb.detectAndCompute(frame, None)
        # match frame descriptors with model descriptors
        # put every match in List match to detect two or three images at the same time.
        try:
            match1 = flann.knnMatch(des1_model,des_frame,k=2)
            match2 = flann.knnMatch(des2_model,des_frame,k=2)
            match3 = flann.knnMatch(des3_model,des_frame,k=2)
            match = [match1, match2, match3]
        except:
            pass
        # store all the good matches as per Lowe's ratio test.
        matches1 = []
        for m,n in match1:
            if m.distance < 0.7*n.distance:
                matches1.append(m)
        matches2 = []
        for m,n in match2:
            if m.distance < 0.7*n.distance:
                matches2.append(m)
        matches3 = []
        for m,n in match3:
            if m.distance < 0.7*n.distance:
                matches3.append(m)
        matches = [matches1, matches2, matches3]
        Matches = {tuple(matches1):70,tuple(matches2):5,tuple(matches3):100}
        # compute Homography if enough matches are found
        for i in range(len(matches)):
            if len(matches[i]) > MIN_MATCHES:
                # differenciate between source points and destination points
                try:
                    a=Homography(kp_model[i],kp_frame,camera_parameters,args,matches[i],model[i],obj[i],frame,matches[i],Matches)
                    cv2.imshow('frame', a)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except:
                    pass
        
            else:
                #print("Not enough matches found - %d/%d" % (len(matches), MIN_MATCHES))
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()
    return 0

def Homography(kp_model,kp_frame,camera_parameters,args,matches,model,obj,frame,match_i,Matches):
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # compute Homography
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if args.rectangle:
        # Draw a rectangle that marks the found model in the frame
        h, w = model.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # project corners into frame
        dst = cv2.perspectiveTransform(pts, homography)
        # connect them with lines  
        frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)  
    # if a valid homography matrix was found render cube on model plane
    if homography is not None:
        # obtain 3D projection matrix from homography matrix and camera parameters
        projection = projection_matrix(camera_parameters, homography)  
        # project cube or model
        frame = render(frame, obj, projection, model,match_i,Matches, True)
        #frame = render(frame, model, projection)
    # draw first 10 matches.
    if args.matches:
        frame = cv2.drawMatches(model, kp_model, frame, kp_frame, matches[:30], 0, flags=2)
    return frame
    # show result
    
    

def render(img, obj, projection, model, match_i, Matches, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    vertices = obj.vertices
    scale_matrix = np.eye(3) * Matches[tuple(match_i)]
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)
        if color is False:
            cv2.fillConvexPoly(img, imgpts, DEFAULT_COLOR)
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(img, imgpts, color)

    return img

def projection_matrix(camera_parameters, homography):
    """
    From the camera calibration matrix and the estimated homography
    compute the 3D projection matrix
    """
    # Compute rotation along the x and y axis as well as the translation
    homography = homography * (-1)
    rot_and_transl = np.dot(np.linalg.inv(camera_parameters), homography)
    col_1 = rot_and_transl[:, 0]
    col_2 = rot_and_transl[:, 1]
    col_3 = rot_and_transl[:, 2]
    # normalise vectors
    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))
    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l
    # compute the orthonormal basis
    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)
    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)
    # finally, compute the 3D projection matrix from the model to the current frame
    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    return np.dot(camera_parameters, projection)

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))


# Command line argument parsing
# NOT ALL OF THEM ARE SUPPORTED YET
parser = argparse.ArgumentParser(description='Augmented reality application')

parser.add_argument('-r','--rectangle', help = 'draw rectangle delimiting target surface on frame', action = 'store_true')
parser.add_argument('-mk','--model_keypoints', help = 'draw model keypoints', action = 'store_true')
parser.add_argument('-fk','--frame_keypoints', help = 'draw frame keypoints', action = 'store_true')
parser.add_argument('-ma','--matches', help = 'draw matches between keypoints', action = 'store_true')
# TODO jgallostraa -> add support for model specification
#parser.add_argument('-mo','--model', help = 'Specify model to be projected', action = 'store_true')

args = parser.parse_args()

if __name__ == '__main__':
    main()  
