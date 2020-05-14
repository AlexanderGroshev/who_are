from flask import Flask, render_template, url_for, redirect, flash, request
from flask_wtf import FlaskForm
from wtforms import TextField, SelectField, StringField
import numpy as np
from PIL import Image
import os
import requests # requests — библиотека Python, которая элегантно и просто выполняет HTTP-запросы 
import pandas as pd
#from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from urllib.request import urlopen
from scipy.spatial import distance
import numpy as np
import os
import face_recognition
import pandas as pd
#from skimage import io
from urllib.request import urlopen
import module as md
from werkzeug.utils import secure_filename


app = Flask(__name__, static_url_path = '/home/ayugroshev/kurs/static')

app.config['IMAGE_UPLOADS'] = r"/home/ayugroshev/kurs/static"

app.config['SECRET_KEY'] = 'amaya'


@app.errorhandler(404)
def http_404_handler(error):
    return render_template('error.html')

@app.errorhandler(500)
def http_500_handler(error):
    return render_template('error.html')


@app.route('/upload_image', methods = ['GET', 'POST'])
def upload_image():

    if request.method == 'POST':

        if request.files:

            image = request.files["image"]

            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename.replace(' ', '')))
            #filename=secure_filename('final_image.jpg')
#            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

           #img = Image.open(r'/home/ayugroshev/kurs/images/' +str(image.filename))
            ur_img =  face_recognition.load_image_file(r'/home/ayugroshev/kurs/static/' +str(image.filename.replace(' ', '')))
            ur_img_encoding = face_recognition.face_encodings(ur_img)
            
            draw_image = md.face_draw('/home/ayugroshev/kurs/static/'+str(image.filename.replace(' ', '')))
            draw_image.save('/home/ayugroshev/kurs/static/draw'+str(image.filename.replace(' ', '')))
            filepath_draw = '/home/ayugroshev/kurs/static/draw'+str(image.filename.replace(' ', ''))
           #resized_img = md.get_img(ur_img)
           #resized_img[0].save('/home/ayugroshev/kurs/static/resized_img.jpg')

            encoded_arr = np.load('saved_en_arr.npy')
            answer = np.load('saved_answer.npy')
            id_arr_encoded = np.load('saved_id.npy')
            
            #face_distances = face_recognition.face_distance(encoded_arr, ur_img_encoding[0])
            face_distances = []
            for i in range (len(encoded_arr)):
                dist = face_recognition.face_distance(encoded_arr[i], ur_img_encoding[0])
                face_distances.append(round(dist[0], 2))

            d1, id1, d2, id2, d3, id3 = md.min3(face_distances)

            percent1=md.percent(d1)
            percent1=str(percent1)

            percent2=md.percent(d2)
            percent2=str(percent2)

            percent3=md.percent(d3)
            percent3=str(percent3)

            #id_answer = face_distances_mean.index(min(face_distances_mean))
            #percent=int(100-min(face_distances_mean)*1000*20/35)
            #percent=str(percent)

            real_url = md.real_pic_url('https://www.hse.ru/org/persons/cimage/' + str(id_arr_encoded[id1]))
            #real_url = 'https://www.hse.ru/org/persons/cimage/'
            real_url_2 = md.real_pic_url('https://www.hse.ru/org/persons/cimage/' + str(id_arr_encoded[id2]))
            real_url_3 = md.real_pic_url('https://www.hse.ru/org/persons/cimage/' + str(id_arr_encoded[id3]))

            morphed_image = md.morph('/home/ayugroshev/kurs/static/'+str(image.filename.replace(' ', '')), real_url)
            morphed_image = Image.fromarray(np.uint8(morphed_image[0]/2 + morphed_image[1]/2))
            morphed_image.save('/home/ayugroshev/kurs/static/morphed'+str(image.filename.replace(' ', '')))
            
            morphed_image2 = md.morph('/home/ayugroshev/kurs/static/'+str(image.filename.replace(' ', '')), real_url_2)
            morphed_image2 = Image.fromarray(np.uint8(morphed_image2[0]/2 + morphed_image2[1]/2))
            morphed_image2.save('/home/ayugroshev/kurs/static/morphed2'+str(image.filename.replace(' ', '')))

            morphed_image3 = md.morph('/home/ayugroshev/kurs/static/'+str(image.filename.replace(' ', '')), real_url_3)
            morphed_image2 = Image.fromarray(np.uint8(morphed_image3[0]/2 + morphed_image3[1]/2))
            morphed_image2.save('/home/ayugroshev/kurs/static/morphed3'+str(image.filename.replace(' ', '')))

            link1 = 'https://www.hse.ru/org/persons/' + str(id_arr_encoded[id1])
            link2 = 'https://www.hse.ru/org/persons/' + str(id_arr_encoded[id2])
            link3 = 'https://www.hse.ru/org/persons/' + str(id_arr_encoded[id3])


            print(real_url)
            print(real_url_2)
            print(real_url_3)
            print(image.filename)
            #print(filename)
            filepath = '/home/ayugroshev/kurs/static/'+str(image.filename.replace(' ', ''))
            filepath_morphed = '/home/ayugroshev/kurs/static/morphed'+str(image.filename.replace(' ', ''))
            filepath_morphed_2 = '/home/ayugroshev/kurs/static/morphed2'+str(image.filename.replace(' ', ''))
            filepath_morphed_3 = '/home/ayugroshev/kurs/static/morphed3'+str(image.filename.replace(' ', ''))
            return render_template('result.html',link1=link1, link2=link2, link3=link3, p1=percent1, p2=percent2, p3=percent3, filepath=filepath, filepath_morphed = filepath_morphed,filepath_morphed_2 = filepath_morphed_2, filepath_morphed_3 = filepath_morphed_3, filepath_draw=filepath_draw, real_url=real_url,real_url_2=real_url_2, real_url_3=real_url_3, name = answer[id1],name_2=answer[id2], name_3=answer[id3])

    return render_template('upload_image.html')

@app.route('/main', methods = ['GET', 'POST'])
def home():
    """
    Функция возвращает шаблон home.html при переходе по адресу /home
    """
    return render_template('main.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 8000)
    
