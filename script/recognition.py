def car_plate(img) :
  import cv2
  import numpy as np
  from google.colab.patches import cv2_imshow
  import IPython  

  img_h, img_w, img_c = img.shape

# YOLO 모델 OpenCV로 불러오기
  weight_file = '/content/yolov3.weights' # weight 파일은 용량이 크므로 코랩에 바로 저장
  cfg_file = '/content/SkillTreePython-DeepLearning/00.추가학습/data/yolov3.cfg'
  name_file = '/content/SkillTreePython-DeepLearning/00.추가학습/data/coco.names'
  model = cv2.dnn.readNet(weight_file, cfg_file) 

# YOLO 모델의 트레인 데이터인 COCO에 사용된 클래스 바인딩
  with open(name_file, 'r') as f :
    class_names = []
    for line in f.readlines() :
      class_names.append(line.strip())  

# 이미지 전처리
# BGR -> RGB
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), 
                              (0,0,0), True, crop=False) # scale : 1/255
  model.setInput(blob) # YOLO 모델에 전처리 모델인 blob 적용

# 계산 (순전파)
  preds = model.forward(['yolo_82','yolo_94','yolo_106'])

  boxes = []
  confidences = []
  class_ids = []

  for pred in preds :
    for v in pred :
      box = v[:4]
      confidence = v[4]

      # 분류한 클래스 이름
      class_id = v[5:] # 80개 배열
      class_id = np.argmax(class_id)
  
      if confidence > 0.8 :
        x_center,y_center,w,h = box
        x_center = int(x_center*img_w)
        y_center = int(y_center*img_h)
        w = int(w*img_w)
        h = int(h*img_h)
        x,y = x_center - int(w/2), y_center - int(h/2)
        
        boxes.append([x,y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
        
  idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  for i, box in enumerate(boxes) :
    class_id = class_ids[i]
    if i in idxs :
      if class_names[class_id] == 'car' :
        x,y,w,h = box
  crop = img[y:y+h, x:x+w]
  gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
  hist = cv2.equalizeHist(gray)
  
  file_plate = '/content/SkillTreePython-DeepLearning/00.추가학습/data/haarcascade_russian_plate_number.xml'
  model = cv2.CascadeClassifier(file_plate)
  preds = model.detectMultiScale(hist)
  for pred in preds :
    x,y,w,h = pred
    cv2_imshow(crop[y:y+h, x:x+w])
    
    
    
    
def img_recognition(img) :
  import cv2
  import numpy as np
  from google.colab.patches import cv2_imshow
  import IPython  

  img_h, img_w, img_c = img.shape

# YOLO 모델 OpenCV로 불러오기
  weight_file = '/content/yolov3.weights' # weight 파일은 용량이 크므로 코랩에 바로 저장
  cfg_file = '/content/SkillTreePython-DeepLearning/00.추가학습/data/yolov3.cfg'
  name_file = '/content/SkillTreePython-DeepLearning/00.추가학습/data/coco.names'
  model = cv2.dnn.readNet(weight_file, cfg_file) 

# YOLO 모델의 트레인 데이터인 COCO에 사용된 클래스 바인딩
  with open(name_file, 'r') as f :
    class_names = []
    for line in f.readlines() :
      class_names.append(line.strip())  

# 이미지 전처리
# BGR -> RGB
  blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), 
                              (0,0,0), True, crop=False) # scale : 1/255
  model.setInput(blob) # YOLO 모델에 전처리 모델인 blob 적용

# 계산 (순전파)
  preds = model.forward(['yolo_82','yolo_94','yolo_106'])
  boxes = []
  confidences = []
  class_ids = []
  colors = np.random.uniform(0,255, (len(class_names), 3))

  for pred in preds :
    for v in pred :
      box = v[:4]
      confidence = v[4]

      # 분류한 클래스 이름
      class_id = v[5:] # 80개 배열
      class_id = np.argmax(class_id)

      if confidence > 0.5 :
        x_center,y_center,w,h = box
        x_center = int(x_center*img_w)
        y_center = int(y_center*img_h)
        w = int(w*img_w)
        h = int(h*img_h)
        x,y = x_center - int(w/2), y_center - int(h/2)       
        boxes.append([x,y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

  idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  for i, box in enumerate(boxes) :
    class_id = class_ids[i]
    color = colors[class_id]
    x,y,w,h = box
    if i in idxs :
      cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
      cv2.putText(img, class_names[class_id], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  cv2_imshow(img)