import os
import pandas as pd
import numpy as np
import face_recognition
import os
from flask import Flask, request, render_template, send_from_directory
import cv2


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))




@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
      
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".jpeg"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")
        destination = "/".join([target, 'new.jpg'])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)

    return render_template("complete.html", image_name=filename ) 




@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images/men", filename)


@app.route('/gallery')
def get_gallery():

    
    dfff = pd.read_csv('men_face_encodings_final.csv', sep=',', encoding='utf-8')
    numpy_rr = dfff.iloc[:, 2:130]
    numpy_r = numpy_rr.astype(float)
    n = numpy_r.to_numpy()

    img = cv2.imread ('images/new.jpg')
    image_to_test_encoding = face_recognition.face_encodings(img)[0]
    face_distances = face_recognition.face_distance(n, image_to_test_encoding)
    dfff['file_name'] = dfff['file'].str.strip(r"C:\Users\User\TEACHME_Homeworks\IMAGES\men")
    dfff['face_dist'] = face_distances
    sorted_df = dfff.sort_values(by='face_dist')
    df_toshow_list_first10 = sorted_df[:10]['file_name'].to_list()
    # df_toshow_list_second10 = sorted_df[10:20]['file_name'].to_list()

    
    return render_template("gallery.html", image_names=df_toshow_list_first10)


if __name__ == "__main__":
    app.run(port=4555, debug=True)



