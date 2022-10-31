import cv2 as cv

model_bin="MobileNetSSD_deploy.caffemodel"
config_text="MobileNetSSD_deploy.prototxt.txt"

objName=["bakground","aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
         "dinigtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]

net=cv.dnn.readNetFromCaffe(config_text,model_bin)
cap=cv.VideoCapture(1)
while True:
    ret,frame=cap.read()
    if ret is False:
        break
    h,w=frame.shape[:2]
    blobImage=cv.dnn.blobFromImage(frame,0.007843,(300,300),(127.5,127.5,127.5),True,False);
    net.setInput(blobImage)
    cvOut=net.forward()
    for detection in cvOut[0,0,:,:]:
        score=float(detection[2])
        objIndex=int(detection[1])
        if score > 0.5:
            left=detection[3]*w
            top=detection[4]*h
            right=detection[5]*w
            bottom=detection[6]*h

            cv.rectangle(frame,(int(left),int(top)),(int(right),int(bottom)),(255,0,0),2,8)
    cv.imshow("video",frame)
    cv.waitKey(0)