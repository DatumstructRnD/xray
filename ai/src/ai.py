from skimage import io
from skimage import measure
import cv2
from PIL import Image
import uuid
import numpy as np
import datetime
import os, glob
import time
import requests
import base64
# import json
# from json import JSONEncoder
from io import BytesIO

import xtect_utils

from mmdet.apis import init_detector, inference_detector
from flask import Flask, make_response, jsonify, Response, request, json

app = Flask(__name__)

class NpArrayEncoder(json.JSONEncoder):
    def default(self,object):
        if isinstance(object,np.ndarray):
            return object.tolist()
        return json.JSONEncoder.default(self,object)

count = 0
current = []
posted_ids = []
color = (255,0,0)

knife_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_knife.py'
knife_checkpoint_file = '/src/model/knife_model_mix1.pth'

gun_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_toygun.py'
gun_checkpoint_file = '/src/model/faster_rcnn_r50_pytorch_datum_toygun.pth'

ied_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_ied.py'
ied_checkpoint_file = '/src/model/faster_rcnn_r50_pytorch_datum_ied.pth'

device = 'cuda:0'
#init a detector
knife_model = init_detector(knife_config_file, knife_checkpoint_file, device=device)
gun_model = init_detector(gun_config_file, gun_checkpoint_file, device=device)
ied_model = init_detector(ied_config_file, ied_checkpoint_file, device=device)


# dir = '/home/dssuser/Desktop/top'
# fileset = os.path.join(dir, '*.jpg')

# filepaths = sorted(glob.glob(fileset))

print('Starting to read files...')
def process_ai(data):
    results = {}

    b64 = data['base64_img']

    im_bytes = base64.b64decode(b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
            #frame = im_file.read()
            #img = Image.open(im_file)   # img is now PIL Image object 
    img = Image.open(im_file)
    img_array = np.array(img)

    result_knife = inference_detector(knife_model, img_array)
    result_toygun = inference_detector(gun_model, img_array)
    result_ied = inference_detector(ied_model, img_array)

    print(result_knife)
    print('')
    print(result_toygun)
    print('')
    print(result_ied)
    print('')

    results['knife'] = result_knife
    results['toygun'] = result_toygun
    results['ied'] = result_ied

    return results

@app.route('/predict', methods=['POST'])
def put_data():
    r = request.get_json()
    data = json.loads(r)

    results = process_ai(data)
    output = json.dumps(results, cls=NpArrayEncoder)

    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)

