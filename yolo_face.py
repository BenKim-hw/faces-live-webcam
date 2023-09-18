import os
import cv2
import torch

import supervision as sv
import numpy as np

from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO

### helper functions
def encode(img, resnet):
    res = resnet(torch.Tensor(img))
    return res

def decode(argv,thres, frame, xyxy, all_people_faces, resnet):
    if len(xyxy)==0: return
    x1, y1, x2, y2 = xyxy.astype(int)
    if y2-y1 < argv.min_size or x2-x1 < argv.min_size:
        return 'Too small'
    cropped = cv2.dnn.blobFromImage(frame[y1:y2,x1:x2,:], 1 / 255.0, (160, 160),swapRB=True, crop=False) 
    img_embedding = encode(cropped, resnet)[0, :]

    detect_dict = {}
    for k, v in all_people_faces.items():
        detect_dict[k] = (v - img_embedding).norm().item()
    min_key = min(detect_dict, key=detect_dict.get)
    print(detect_dict)

    if detect_dict[min_key] >= thres:
        min_key = 'Unknown'

    text = min_key
    return text

def get_closest_text(xyxy, text_dic):
    text_pos_arr = [np.array(i) for i in text_dic.keys()]
    text_min_dist = list(text_dic.values())[np.argmin(np.linalg.norm(text_pos_arr - xyxy, axis=1))]
    return text_min_dist


def draw_text(frame, xyxy_arr, text_dic, box_annotator):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(xyxy_arr)):
        x1, y1, x2, y2 = xyxy_arr[i].astype(int)
        text = get_closest_text(xyxy_arr[i], text_dic)

        text_width, text_height = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=box_annotator.text_scale,
            thickness=box_annotator.text_thickness,
        )[0]

        text_x = x1 + box_annotator.text_padding
        text_y = y2 + box_annotator.text_padding + text_height

        cv2.rectangle(
            img=frame,
            pt1=(x1, y2),
            pt2=(x1 + 2 * box_annotator.text_padding + text_width,
                y2 + 2 * box_annotator.text_padding + text_height),
            color=(255,255,255),
            thickness=cv2.FILLED,
        )

        cv2.putText(
            img=frame,
            text=text,
            org=(text_x, text_y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=box_annotator.text_scale,
            color=box_annotator.text_color.as_rgb(),
            thickness=box_annotator.text_thickness,
            lineType=cv2.LINE_AA,
        )


def run(argv):
    ### Load models
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    model = YOLO('./models/yolov8n-face.pt')
    print('Finished loading models')

    ### get encoded features for all saved images
    saved_pictures = "./saved/"
    all_people_faces = {}
    thres = argv.close_thr

    for person_face, extension in [i.split('.') for i in os.listdir(saved_pictures) if not i.startswith('.')]:
        img = cv2.imread(f'{saved_pictures}/{person_face}.{extension}')
        result = model(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > argv.face_thr]
        if len(detections) == 0:
            print(f'ERROR: No faces in image for {person_face}')
            continue
        max_idx = np.argmax(detections.confidence)
        x1, y1, x2, y2 = detections.xyxy[max_idx].astype(int)
        cropped = cv2.dnn.blobFromImage(img[y1:y2,x1:x2,:], 1 / 255.0, (160, 160),swapRB=True, crop=False) 
        all_people_faces[person_face] = encode(cropped, resnet)[0, :]

    print('Finished loading faces')

    ### to save the video
    # writer= cv2.VideoWriter('webcam_yolo.mp4', 
    #                         cv2.VideoWriter_fourcc(*'DIVX'), 
    #                         7, 
    #                         (1280, 720))
    
    # define resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # customize the bounding box
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    i = 0
    while True:
        _, frame = cap.read()
        result = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > 0.5] # confidence thr for live image
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _ in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        ) 

        if i % argv.frames_label == 0:
            text_dic = {}
            if len(detections.xyxy)>0:
                for xyxy in detections.xyxy:    
                    text = decode(argv, thres, frame, xyxy, all_people_faces, resnet)
                    text_dic[tuple(xyxy)] = text
            i = 0

        if len(detections.xyxy)>0:
            if len(detections.xyxy) == len(text_dic):
                draw_text(frame, detections.xyxy, text_dic, box_annotator)

        cv2.imshow("face-webcam", frame)
        
        i += 1
        if (cv2.waitKey(10) == 27): # break with escape key
            break

    cap.release()
    # writer.release()
    cv2.destroyAllWindows()
