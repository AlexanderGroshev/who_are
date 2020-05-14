import requests
import dlib
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
from urllib.request import urlopen


def percent(distance):
    if (distance<=0.55):
        return int(round(2-np.exp(distance/3), 2)*100)
    elif (distance>0.55) and (distance<0.8):
        return int(round(np.exp((-distance+0.55)*4)-0.21, 2)*100)
    else:
        return 10

def real_pic_url(url):
    try:
        response = requests.get(url)
        if response.history:
            # Request was redirected
            return response.url
        else:
            # Request was not redirected"
            return url
    except Exception as e:
        return False
        pass

def resize_image(input_image_path,
                 size):
    #original_image = Image.open(input_image_path)
    original_image = input_image_path
    width, height = original_image.size
    resized_image = original_image.resize(size)
    width, height = resized_image.size
    return resized_image

def get_img (image):
    face_location = face_recognition.face_locations(image)
    top, right, bottom, left = face_location[0]

    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    resized_pil_image = resize_image(input_image_path= pil_image, size=(250, 250))
    
    return resized_pil_image, np.asarray(resized_pil_image)

def morph (path1, path2):
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    img1 = dlib.load_rgb_image(path1)
    img2 = np.array(Image.open(urlopen(path2)))
    
    dets1 = detector(img1, 1)
    dets2 = detector(img2, 1)
    
    faces1 = dlib.full_object_detections()
    faces2 = dlib.full_object_detections()
    
    for detection in dets1:
        faces1.append(sp(img1, detection))
    
    for detection in dets2:
        faces2.append(sp(img2, detection))
    
    image1 = dlib.get_face_chip(img1, faces1[0], size= 300)
    image2 = dlib.get_face_chip(img2, faces2[0], size= 300)
    
    return np.array(image1), np.array(image2)

def face_draw(path):
    image = face_recognition.load_image_file(path)

    face_landmarks_list = face_recognition.face_landmarks(image)
    face_location = face_recognition.face_locations(image)
    top, right, bottom, left = face_location[0]
    
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    d.line([left,top,right,top], width=10)
    d.line([left,bottom,right,bottom], width=10)
    d.line([left,bottom,left,top], width=10)
    d.line([right,bottom,right,top], width=10)

    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)
        
    return pil_image


def min3(array):
    m1 = min(array)
    id1 = array.index(min(array))
    for j in range (len(array)):
        if (array[j] == m1):
            array[j] = 1
    m2 = min(array)
    id2 = array.index(min(array))
    for j in range (len(array)):
        if (array[j] == m2):
            array[j] = 1
    m3 = min(array)
    id3 = array.index(min(array))
    return m1, id1, m2, id2, m3, id3
