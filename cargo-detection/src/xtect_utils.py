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

def get_unique_cargo(current, img, count):

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres,bin = cv2.threshold(grey,200,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((7,7),np.uint8)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, kernel)
    bin = cv2.morphologyEx(bin, cv2.MORPH_OPEN, kernel)
    bin_img = Image.fromarray(bin).crop((195,40,1725,1000))
    bin_array = np.array(bin_img)

    label = measure.label(bin_array)
    regions = measure.regionprops(label)

    detections = []
    for region in regions:
        detection = {}
        bbox = region.bbox
        area = region.area
        detection['bbox'] = bbox
        detection['area'] = area
        detection['centroid'] = [(bbox[3]+bbox[1])/2,(bbox[2]+bbox[0])/2]
        if (detection['area'] > 800): 
            detections.append(detection)
    
    sorted_detections = sorted(detections,key=lambda d: d['area'],reverse=False)

    cargo = []
    for i in range(len(sorted_detections)):
        # print("i = " + str(i))
        test=[]
        for j in range(i+1,len(sorted_detections)):
            # print("j= " +str(j))
            is_within_x = (sorted_detections[j]['bbox'][1]) <= (sorted_detections[i]['centroid'][0]) <= sorted_detections[j]['bbox'][3]
            is_within_y = (sorted_detections[j]['bbox'][0]) <= (sorted_detections[i]['centroid'][1]) <= sorted_detections[j]['bbox'][2]
            test.append(is_within_x & is_within_y)
            #test.append(is_within_y)
            # print(is_within_x)
            # print(is_within_y)
            
        #print(test)
        if any(test):
            pass
        else:
            cargo.append(sorted_detections[i])
    
    #remove = []
    for cur in current:
        distances = []
        id = []
        for car in cargo: 
            distance = np.linalg.norm(np.array(cur['centroid'])-np.array(car['centroid']))
            distances.append(distance)
        
        index = np.argmin(distances)
        if cargo[index]['centroid'][0] <= (cur['centroid'][0]+20):
            cargo[index]['id'] = cur['id']
        # else:
        #     remove.append(index)

    # for index in remove:
    #     del current[index]
        

    for c in cargo:
        if 'id' in c:
            pass
        else:
            c['id'] = count +1
            count +=1
    
    return cargo, count

def scale_cargo(cargo,img):
    num_window = 1
    
    bbox = cargo['bbox']
    left = (bbox[1])+195
    top = (bbox[0])+40
    right = (bbox[3])+195
    bottom = (bbox[2])+40
    #imgWidth, imgHeight,_ = img.shape

    crop_img = Image.fromarray(img).crop((left,top,right,bottom))
    cropWidth, cropHeight = crop_img.size
    
    if (cropWidth<1000) & (cropHeight<310):
        print('Small object...')
        num_window = 1
        imgs_array_padded = []
        crop_resize_img = crop_img.resize((int(cropWidth*1.8),int(cropHeight*3.4)))
        resizeWidth, resizeHeight = crop_resize_img.size
        img_array = np.array(crop_resize_img)
        border_width = ((1920-resizeWidth)//2)
        border_height = ((1080-resizeHeight)//2)
        img_array_padded = cv2.copyMakeBorder(img_array, border_height, border_height,border_width, border_width, cv2.BORDER_CONSTANT, value = (255,255,255))
        img_array_padded=cv2.resize(img_array_padded,(1920,1080))
        imgs_array_padded.append(img_array_padded)
        return imgs_array_padded, [border_width,border_height], [num_window, num_window]
    
    else:
        print('Big object...')
        num_vertical_window = (cropHeight//300) + 1 
        num_horizontal_window = (cropHeight//1000) + 1
       
        borders = []
        imgs_array_padded = []

        for i in range(num_horizontal_window):
            for j in range(num_vertical_window): 
                sub_left = i*1000
                sub_top = j*300
                sub_right = np.min([sub_left+1000,cropWidth])
                sub_bottom = np.min([sub_top+300,cropHeight])
                print(sub_left,sub_top,sub_right,sub_bottom)
                sub_crop = crop_img.crop((sub_left,sub_top,sub_right,sub_bottom))
                sub_crop_width, sub_crop_height = sub_crop.size
                sub_crop_resize_img = sub_crop.resize((int(sub_crop_width*1.8),int(sub_crop_height*3.4)))
                resizeWidth, resizeHeight = sub_crop_resize_img.size
                print(resizeWidth)
                print(resizeHeight)
                img_array = np.array(sub_crop_resize_img)
                border_width = ((1920-resizeWidth)//2)
                border_height = ((1080-resizeHeight)//2)
                print(border_width)
                print(border_height)
                img_array_padded = cv2.copyMakeBorder(img_array, border_height, border_height,border_width, border_width, cv2.BORDER_CONSTANT, value = (255,255,255))
                img_array_padded=cv2.resize(img_array_padded,(1920,1080))
                
                borders.append([border_width,border_height])
                imgs_array_padded.append(img_array_padded)

        return imgs_array_padded, borders, [num_horizontal_window, num_vertical_window]
    # else:
    #     is_multi_window = True
    #     return img, [0,0], num_window


    
def transform_res(bbox, border,result, offset=[0,0]):
    print('transforming res...')
    x_ratio = 1.8
    y_ratio = 3.4

    offset_x = offset[0]
    offset_y = offset[1]

    output = []
    print(result)
    print(len(result))
    for res in result:
        res = np.squeeze(res)
        #print(res)
        #print(res.shape)
        if res.any():
            print('shape' + str(res.shape[0]))
            if len(res.shape)!= 1:
                for det in res:
                    # det = np.squeeze(det)
                    #print(det)
                    left = ((int(det[0]) - border[0]) / x_ratio) + bbox[1] + 195 + offset_x
                    top = ((int(det[1]) - border[1]) / y_ratio) + bbox[0] + 40 + offset_y
                    right = ((int(det[2]) - border[0]) / x_ratio) + bbox[1] + 195 + offset_x
                    bottom = ((int(det[3]) - border[1]) / y_ratio) + bbox[0] + 40 + offset_y
                    label_prob = det[4]
                    array = np.array([left,top,right,bottom,label_prob])
                    array = np.expand_dims(array,0)
                    output.append(array)
            else: 
                    det = res
                    #print(det)
                    left = ((int(det[0]) - border[0]) / x_ratio) + bbox[1] + 195 + offset_x
                    top = ((int(det[1]) - border[1]) / y_ratio) + bbox[0] + 40 + offset_y
                    right = ((int(det[2]) - border[0]) / x_ratio) + bbox[1] + 195 + offset_x
                    bottom = ((int(det[3]) - border[1]) / y_ratio) + bbox[0] + 40 + offset_y
                    label_prob = det[4]
                    array = np.array([left,top,right,bottom,label_prob])
                    array = np.expand_dims(array,0)
                    output.append(array)

    if output:
        return np.stack(np.array(output),0)
    else:
        return np.expand_dims(result,0)
        
def create_json_predict(img):
    # imageid = str(uuid.uuid4())

    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode('utf-8')

    data =  {            
            "base64_img" : im_b64,
            }

    j = json.dumps(data)

    return j

def create_json_put(cargo,result,img,id):
    imageid = str(uuid.uuid4())

    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    im_b64 = im_b64.decode('utf-8')

    uuid_group = []
    to_flag = False

    labels = ['knife', 'toygun', 'IED']

    for c in cargo:
        uid = {}
        #print(c['id'])
        uid['uuid'] = c['id']
        
        c_bbox = c['bbox']
        c_left = (c_bbox[1])+195
        c_top = (c_bbox[0])+40
        c_right = (c_bbox[3])+195
        c_bottom = (c_bbox[2])+40    
        uid['bbox'] = [c_left,c_top,c_right,c_bottom]
        i=0

        detections = []
        for det_class in result:
            label = labels[i]
            #print(label)
            # print(len(det_class))
            # print(det_class)
            for res in det_class:
                # print(len(res))
                # print(res)
                for det in res:
                    #print(det)
                    if det.any():
                        detection = {}
                        det = np.squeeze(det)
                        left = int(det[0])
                        top = int(det[1])
                        right = int(det[2])
                        bottom = int(det[3])
                        label_prob = det[4]
                        centroid = [(left+right)/2, (top+bottom)/2]
                        # is_belong_cargo = (c_left<=centroid[0]<=c_right) & (c_top<=centroid[1]<=c_bottom)
                        is_detection = (label_prob > 0.4) & (bottom < 930) & (top > 50) 
                        if is_detection:
                            detection['label'] = label
                            detection['detection_box'] = [left,top,right,bottom]
                            detection['score'] = str(label_prob)
                            detections.append(detection)
                            to_flag = True       
                            print("label_prob: ", label_prob)
            i +=1
        
        uid['detections'] = detections
        uid['detection_no'] = len(detections)
        uuid_group.append(uid)
    
    cargo_len = len(uuid_group)

    data =  {            
            "image_id" : imageid,
            "focus_id" : id,
            "machine_id" : "T192201F",
            "machine_name" : "Machine 508",
            "machine_type" : "Cargo",
            "base64_img" : im_b64,
            "uuid_group" : uuid_group,
            "uuid_no" : cargo_len,
            "to_flag" : to_flag
            }

    #print(data)
    j = json.dumps(data)
    # with open('json_data.json', 'w') as outfile:
    # 	outfile.write(j)
    
    #r = requests.post(put_uri, json=j, verify='cert.pem')
    return j

# def draw_uid(bboxes, img):
#     for item in bboxes:
#         bbox = item['bbox']
#         left = (bbox[1])+195
#         top = (bbox[0])+40
#         right = (bbox[3])+195
#         bottom = (bbox[2])+40
#         imgWidth, imgHeight,_ = img.shape
#         thick = int((imgHeight + imgWidth) // 900)
#         #print(left, top, right, bottom)
#         id = str(item['id'])
        
#         cv2.rectangle(img,(left, top), (right, bottom), color, thick)
#         cv2.putText(img, id, (left, top - 12), 0, 5e-4 * imgHeight, color, thickness=2)

# def draw_mmdet(imageData, inferenceResults):
#     """Draw bounding boxes on an image.
#     imageData: image data in numpy array format
#     imageOutputPath: output image file path
#     inferenceResults: inference results array off object (l,t,w,h)
#     colorMap: Bounding box color candidates, list of RGB tuples.
#     """
#     labels = ['knife', 'toygun', 'IED']
#     colors = [(255,0,0),(0,255,0),(0,0,255)]

#     i=0
#     for detection_class in inferenceResults:
#         label = labels[i]
#         color = colors[i]
#         for res in detection_class:
#             if res.any():
#                 for detection in res:
#                     detection = np.squeeze(detection)
#                     left = int(detection[0])
#                     top = int(detection[1])
#                     right = int(detection[2])
#                     bottom = int(detection[3])
#                     label_prob = detection[4]
#                     if (label_prob > 0.4) & (bottom < 930) & (top > 50) :
#                         imgHeight, imgWidth, _ = imageData.shape
#                         thick = int((imgHeight + imgWidth) // 900)
#                         #print(left, top, right, bottom)
#                         cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
#                         text = label + "|" + str(round(label_prob,2))
#                         cv2.putText(imageData, text, (left, top - 12), 0, 1e-3 * imgHeight, color, thickness=2)
#         i +=1
    
#     #return imageData
