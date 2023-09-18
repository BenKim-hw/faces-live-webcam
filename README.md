# Faces-live-webcam
Live facial recognition and classification from your webcam using pretrained yolov8 and facenet models and openCV

The code is based on publically available models [yolov8-face](https://github.com/akanametov/yolov8-face) and [facenet](https://github.com/davidsandberg/facenet). The [pytorch implementation](https://github.com/timesler/facenet-pytorch) of facenet was used.

# About
The model uses yolov8-face to detect faces in the webcam frames. It then extracts features of faces using facenet, and compare it with features from saved images. If the similiarity is above a threshold, the face is labeled. If no saved images show similiarity above the threshold, the face is labeled 'Unknown'. 

# How to use
## Initialize
```
git clone https://github.com/BenKim-hw/faces-live-webcam.git
cd faces-live-webcam
pip install -r requirements.txt
```

## Model weights
Pretrained model weights for yolov8-face can be downloaded [here](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt). Download the weights under the models/ folder.
```
wget https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt -P models/
```

## Labeled faces
Place your labeled faces under the saved/ folder. The name of the images will be displayed as labels. 

## Directory tree
```bash
├── models
│   ├── yolov8n-face.pt
├── saved
│   ├── label1.jpg
│   ├── label2.jpg
├── utils
│   ├── __init__.py
│   └── parse.py
├── main.py
└── yolo_face.py

```

## Run
```
python main.py --face_thr 0.5 --close_thr 1.0 --min_size 80 --frames_label 5
```

- -t, --face_thr = Lower threshold of the probability of the detected face being a face under **yolov8-face**. *Default value* = 0.5
- -u, --close_thr = Upper threshold of the closeness (l2 norm) between **facenet embeddings** of labeled (saved) images and the detected face. *Default value* = 1.0
- -m, --min_size = Images under the size of min_size x min_size will not be labeled and displayed "Too small". If this value is too small, it can cause errors due to the input size limit of facenet. *Default value* = 80, *recommended value* ≥ 50
- -f, --frames_label = The number of frames that the labels will be updated. If this value is too small, it may cause frame drops. *Default value* = 5

# License
The models were relased under the following licenses:
- yolov8-face: GPL-3.0 License
- facenet: MIT License
