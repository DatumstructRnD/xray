from distutils.log import debug
import os
import glob
import time
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import datetime

from flask import Flask, make_response, jsonify, Response, request

app = Flask(__name__)

try:
    draw_mode = int(os.environ.get('DETMODE'))
except:
    draw_mode = 7

current = {}
color = {'knife':(255,0,125), 'toygun':(126,255,255), 'IED':(0,0,255)}

"""
    This is edited version haha
"""

fname = '/src/logo/logo-600x315.png'
with open(fname, "rb") as image:
    frame = image.read()

def gen():
    global current
    global frame
    while True:
        try:
            img_array = draw() 

            # io_buf = BytesIO(img_array)
            # frame = io_buf.read()

            is_success, im_buf_arr = cv2.imencode(".jpg", img_array)
            byte_im = im_buf_arr.tobytes()
            im_file = BytesIO(byte_im)
            frame = im_file.read()
                  
        except Exception as e:
            print(e)
            pass

        time.sleep(0.1)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def draw():
    b64 = current['base64_img']

    im_bytes = base64.b64decode(b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
            #frame = im_file.read()
            #img = Image.open(im_file)   # img is now PIL Image object 
    img = Image.open(im_file)
    img_array = np.array(img)

    cv2.putText(img_array, current['machine_name'], (10, 30), 0, 1, (255,255,255), thickness=2)

    uuid_group = current['uuid_group']
    for uuid in uuid_group:
        if uuid['uuid'] == current['focus_id']:
            id = str(uuid['uuid'])
            bbox = uuid['bbox']
            left = (bbox[0])
            top = (bbox[1])
            right = (bbox[2])
            bottom = (bbox[3])
            imgWidth, imgHeight,_ = img_array.shape
            thick = int((imgHeight + imgWidth) // 1800)
                    
            cv2.rectangle(img_array,(left-10, top-10), (right+10, bottom+10), (128,0,0), thick)
            #cv2.putText(img_array, id, (left, top - 12), 0, 5e-4 * imgHeight, (255,0,0), thickness=2)
                    
            for detection in uuid['detections']:
                if(draw_mode==1):
                    if(detection['label']=='knife'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==2):
                    if(detection['label']=='toygun'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==3):
                    if(detection['label']=='knife'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                    if(detection['label']=='toygun'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==4):
                    if(detection['label']=='IED'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,5e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==5):
                    if(detection['label']=='knife'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                    if(detection['label']=='IED'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,5e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==6):
                    if(detection['label']=='toygun'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,5e-4*imgHeight,color[detection['label']],thickness=1)
                    if(detection['label']=='IED'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,5e-4*imgHeight,color[detection['label']],thickness=1)
                if(draw_mode==7):
                    if(detection['label']=='knife'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                    if(detection['label']=='toygun'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,2e-4*imgHeight,color[detection['label']],thickness=1)
                    if(detection['label']=='IED'):              
                        bbox_detection = detection['detection_box']
                        left = (bbox_detection[0])
                        top = (bbox_detection[1])
                        right = (bbox_detection[2])
                        bottom = (bbox_detection[3])
                        imgWidth, imgHeight,_ = img_array.shape
                        thick = int((imgHeight + imgWidth) // 900)

                        text = detection['label'] + " | " + str(round(float(detection['score']),2))
                        cv2.rectangle(img_array,(left,top),(right,bottom),(0,0,255), thick)
                        cv2.putText(img_array,text,(left,top-12),0,5e-4*imgHeight,color[detection['label']],thickness=1)				
    return img_array


        

@app.route('/send', methods=['POST'])
def put_data():
    global current
    r = request.get_json()
    #print(r)
    current = json.loads(r)

    img_array = draw()

    save_file = img_array.copy()
    now = datetime.datetime.now()
    filename = "/src/results/" + str(now) + '.jpg'
    cv2.imwrite(filename, save_file)
    
    res = make_response(jsonify({"message": "OK"}))
    return res

@app.route('/video', methods=['GET'])
def get_video():
    
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=True)
