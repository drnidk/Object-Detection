import cv2

classFile = "/Users/darrenso/codingstuff/coco (1).names"
configPath = "/Users/darrenso/codingstuff/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
paths = "/Users/darrenso/codingstuff/frozen_inference_graph (1).pb"

with open (classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print (classNames)

net = cv2.dnn_DetectionModel(paths,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean([127.5, 127.5, 127.5])
net.setInputSwapRB(True)

print(net)

#Set the drawn box dimensions and color to adjust according to the size of the object

def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres,nmsThreshold=nms)
    #print (classIds.bbox)
    if len(objects) == 0: objects = classNames
    objectsInfo = []
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):

            className = classNames[classId - 1]
            if className in objects:
                objectsInfo.append([box,className])
                if(draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    return img,objectsInfo

#add video file

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set (3,640)
    cap.set (4,480)
    while True:
     success, img = cap.read()
     result, objectInfo = getObjects(img,0.45,0.2)
     cv2.imshow("Output",img)
     cv2.waitKey(1)

     if cv2.waitKey(1) & 0xFF == ord('q'):
         break

