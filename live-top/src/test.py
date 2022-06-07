import cv2
import time
import datetime
import os 
import glob 
import numpy as np

# from mmdet.apis import init_detector, inference_detector

# knife_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_knife.py'
# knife_checkpoint_file = '/src/model/faster_rcnn_r50_pytorch_datum_knife.pth'

# gun_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_toygun.py'
# gun_checkpoint_file = '/src/model/faster_rcnn_r50_pytorch_datum_toygun.pth'

# ied_config_file = '/src/model/faster_rcnn_r50_pytorch_datum_ied.py'
# ied_checkpoint_file = '/src/model/faster_rcnn_r50_pytorch_datum_ied.pth'

# device = 'cuda:0'
# #init a detector
# knife_model = init_detector(knife_config_file, knife_checkpoint_file, device=device)
# gun_model = init_detector(gun_config_file, gun_checkpoint_file, device=device)
# ied_model = init_detector(ied_config_file, ied_checkpoint_file, device=device)

def drawBoundingBoxes(imageData, imageOutputPath, inferenceResults):
    """Draw bounding boxes on an image.
    imageData: image data in numpy array format
    imageOutputPath: output image file path
    inferenceResults: inference results array off object (l,t,w,h)
    colorMap: Bounding box color candidates, list of RGB tuples.
    """
    labels = ['knife', 'toygun', 'IED']
    colors = [(255,0,0),(0,255,0),(0,0,255)]

    i=0
    for detection_class in inferenceResults:
        label = labels[i]
        color = colors[i]
        for res in detection_class:
            if res.any():
                for detection in res:
                    detection = np.squeeze(detection)
                    left = int(detection[0])
                    top = int(detection[1])
                    right = int(detection[2])
                    bottom = int(detection[3])
                    label_prob = detection[4]
                    if (label_prob > 0.4) & (bottom < 930) & (top > 50) :
                        imgHeight, imgWidth, _ = imageData.shape
                        thick = int((imgHeight + imgWidth) // 900)
                        #print(left, top, right, bottom)
                        cv2.rectangle(imageData,(left, top), (right, bottom), color, thick)
                        text = label + "|" + str(round(label_prob,2))
                        cv2.putText(imageData, text, (left, top - 12), 0, 1e-3 * imgHeight, color, thickness=2)
        i +=1
    
    return imageData
    #cv2.imshow('top',imageData)

try:
    is_RGB = os.environ.get('RGB')
except:
    is_RGB = 'True'
    pass


current_time = time.time()
last_time = current_time

is_cap_opened = False
while not is_cap_opened:
    cap = cv2.VideoCapture(0)
    is_cap_opened = cap.isOpened()
    if not is_cap_opened:
        print("Cannot open camera")
        time.sleep(1)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #print(frame.shape)
    frame=cv2.resize(frame,(1920,1080))

    if is_RGB == 'False':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    now = datetime.datetime.now()
    # result = inference_detector(knife_model, frame)
    # print(result)

    # result = []
    # result_knife = inference_detector(knife_model, frame)
    # #print(result_knife)
    # result_toygun = inference_detector(gun_model, frame)
    # result_ied = inference_detector(ied_model, frame)

    # result.append(result_knife)
    # result.append(result_toygun)
    # result.append(result_ied)

    # filename = "./side/" + str(now) + '.jpg'

    #knife_model.show_result(frame,result, out_file=filename)

    #output = drawBoundingBoxes(frame,filename,result)
    #cv2.imwrite(filename, frame)
    #knife_model.show_result(frame,result, out_file=filename)
    filename = "/src/images/" + str(now) + '.jpg'
    #knife_model.show_result(frame,result, out_file=filename)
    #cv2.imshow('results',image_out)
    current_time = time.time()
    if (current_time - last_time) >0.5:
        cv2.imwrite(filename, frame)
        list_of_files = glob.glob('/src/images/*jpg') 
        if len(list_of_files) > 100:
            oldest_file = min(list_of_files, key=os.path.getctime)
            os.remove(oldest_file)
        last_time = time.time()
    #frame = drawBoundingBoxes(frame,filename,result)
    cv2.namedWindow('Top',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Top',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #cv2.moveWindow('Side',1920,0)
    cv2.imshow('Top',frame)
    cv2.waitKey(1)

    # list_of_files = glob.glob('/src/images/*jpg') 
    # if len(list_of_files) > 50:
    #     oldest_file = min(list_of_files, key=os.path.getctime)
    #     os.remove(oldest_file)


