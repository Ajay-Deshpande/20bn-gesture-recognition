import joblib
from glob import glob
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json



if __name__ == "__main__":

    json_file = open(glob('./model/*.json')[0], 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(glob('./model/*.h5')[0])

    camera = cv2.VideoCapture(0)

    classes = np.load(glob("./model/*encoder*.npy")[0],allow_pickle=True).tolist()[0]
    # classes = ['No gesture','Stop Sign','Swiping Down','Swiping Left','Swiping Right','Swiping Up','Zooming In With Full Hand','Zooming Out With Full Hand']
    num_frames = 0
    li = []
    def reset():
        global num_frames
        global li
        num_frames = 0
        li = []
    label = "no action"
    cooldown = False

    while(True):
        (grabbed, frame) = camera.read()
        frame = cv2.flip(frame, 1)
        (height, width) = frame.shape[:2]
        right = int(width/2 + height/2)
        left = int(width/2 - height/2)
        top = 0
        bottom = height
        inp = frame[top:bottom, left:right]
        inp = cv2.resize(inp , (32,32))

        if num_frames < 36:
            li.append(inp)
        
        if num_frames >= 36:
            li.pop(0)
            li.append(inp)
        
        if (num_frames > 36) and ((num_frames-36) % 15) == 0 :
            inp_video = np.asarray(li,dtype=np.float64)[None,:]
            out = np.argsort(loaded_model.predict(inp_video, verbose = 1, use_multiprocessing = 1)[0])
            label = classes[out[-1]]
            reset()
            
        cv2.rectangle(frame, (int(left), top), (int(right), bottom), (0,255,0), 2)
        num_frames += 1
        text = "activity: {}".format(label)
        cv2.putText(frame, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

        cv2.imshow("Video Feed", frame)
        keypress = cv2.waitKey(1) & 0xFF

        if keypress == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()