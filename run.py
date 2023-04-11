import cv2
import numpy as np
import torch
from model import Conv2d_cd, CDCN, CDCNpp
import time

path = 'haarcascade_frontalface_alt2.xml'
face_cascade = cv2.CascadeClassifier(path)
device = torch.device("cuda:0")
model = CDCNpp()
model = model.to(device)
model.load_state_dict(torch.load('CDCNpp_P4_290.pkl'))
model.eval()
cap = cv2.VideoCapture(0)
count = 0
while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
    f = None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        f = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if f is not None:
        f = cv2.resize(f, (256, 256))
        # to tensor
        f = f[:, :, ::-1].transpose((2, 0, 1))
        f = f[np.newaxis, :, :, :]
        f = np.array(f)
        f = torch.from_numpy(f.astype(np.float)).float()
        # normalization
        f = (f - 127.5) / 128
        f = f.cuda()
        map_x = model(f)
        # map_x = map_x.detach().numpy()
        # map_x = map_x.transpose(1, 2, 0)
        sum = torch.sum(map_x)
        count = 0
        if sum >= 325:
            print(sum.item(), " true")
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            name = 'Real'
        else:
            print(sum.item(), " false")
            color = (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            name = 'Spoof'
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # cv2.imshow("video", img)
        # cv2.waitKey(3500)
        del f

    cv2.imshow("video", img)
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
