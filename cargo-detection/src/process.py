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
import json
import xtect_utils

try:
    host_ip = os.environ.get('HOSTIP')
except:
    raise Exception('No host IP defined')

put_uri = 'http://' + str(host_ip) + ':5001/send'
predict_uri = 'http://' + str(host_ip) + ':5000/predict'

count = 0
current = []
posted_ids = []
color = (255,0,0)

print('Starting to read files...')
while True:
    try:
        list_of_files = glob.glob('/src/images/*.jpg')
        latest_file = max(list_of_files, key=os.path.getctime)
        # for latest_file in list_of_files:
        frame = io.imread(latest_file)

        frame=cv2.resize(frame,(1920,1080))

        img = frame.copy()
        input = frame.copy()

        # print('getting unique cargo')
        cargo, count = xtect_utils.get_unique_cargo(current, img, count)
        # print('unique cargo gotten')
        current = cargo

        for cargo in current:
            x, y = cargo['centroid']
            is_center = 400<x<1500
            is_not_edge = cargo['bbox'][3] < 1450
            #print(cargo['bbox'])
            is_not_posted = (cargo['id'] not in posted_ids) 
            # is_not_posted = True
            if is_center & is_not_posted & is_not_edge:

                result = []                
                print('Scaling image...')
                input_scaled, border, num_window = xtect_utils.scale_cargo(cargo,input)
                
                if len(input_scaled) == 1:
                    # print('Running inference...')

                    j = xtect_utils.create_json_predict(input_scaled[0])
                    r = requests.post(predict_uri, json=j)
                    r = r.json()

                    result_knife = np.asarray(r['knife'])
                    result_toygun = np.asarray(r['toygun'])
                    result_ied = np.asarray(r['ied'])
                
                    result_knife = xtect_utils.transform_res(cargo['bbox'],border,result_knife)
                    result_toygun = xtect_utils.transform_res(cargo['bbox'],border,result_toygun)
                    result_ied = xtect_utils.transform_res(cargo['bbox'],border,result_ied)
                
                else:
                    max_window_x = 1000
                    max_window_y = 300

                    result = []
                    result_knife = []
                    result_toygun = []
                    result_ied = []

                    n = 0
                    for i in range(num_window[0]):
                        for j in range(num_window[1]):
                            print('Running inference for %dth window...' %(n))
                            offset_x = i*1000
                            offset_y = j*300

                            sub_input = input_scaled[n]
                            sub_border = border[n]
                            
                            j = xtect_utils.create_json_predict(sub_input)
                            r = requests.post(predict_uri, json=j)
                            r = r.json()

                            result_knife = np.asarray(r['knife'])
                            result_toygun = np.asarray(r['toygun'])
                            result_ied = np.asarray(r['ied'])
                        
                            sub_result_knife = xtect_utils.transform_res(cargo['bbox'],sub_border,sub_result_knife,[offset_x,offset_y])
                            sub_result_toygun = xtect_utils.transform_res(cargo['bbox'],sub_border,sub_result_toygun,[offset_x,offset_y])
                            sub_result_ied = xtect_utils.transform_res(cargo['bbox'],sub_border,sub_result_ied,[offset_x,offset_y])

                            try:
                                result_knife.extend(sub_result_knife)
                                result_toygun.extend(sub_result_toygun)
                                result_ied.extend(sub_result_ied)
                            except:
                                pass

                            n +=1
                
                result.append(result_knife)
                result.append(result_toygun)
                result.append(result_ied)

                print(len(result))
                print(result[0])
                print(result[1])
                print(result[2])
                
                print('Sending request...')
                j = xtect_utils.create_json_put(current,result,input,cargo['id'])
                r = requests.post(put_uri, json=j)
                
                posted_ids.append(cargo['id'])
                
                for img in input_scaled:
                    save_file = img
                    now = datetime.datetime.now()
                    filename = "/src/archive/" + str(now) + '.jpg'
                    save_file = cv2.cvtColor(save_file, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(filename, save_file)

    except Exception as err:
        #print('Unable to read file!')
        print(str(err))
        pass
