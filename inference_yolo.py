import cv2
# import numpy as np
from ultralytics import YOLO
# from helper import create_video_writer
# from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial import distance
from time import time_ns
import pandas as pd
start = time_ns()
model = YOLO('best_move_2.pt')
# tracker = DeepSort(max_age=1)
CONFIDENCE_THRESHOLD = 0.57
VIDEO_FILENAME = r"C:\Users\user\Downloads\first_particle\first_particle.avi"
NEW_TRESHOLD = 100
FAR_TRESHOLD = 0.027
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
# Create a VideoCapture object
video_cap = cv2.VideoCapture(VIDEO_FILENAME)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(video_cap.get(3))
frame_height = int(video_cap.get(4))

def compute_center_coords(x1, y1, x2, y2):
    return ((x1 + x2) / 2), ((y1 + y2) / 2)

def get_video_duration(video):

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    return (frame_count / fps) * 1000_000_000, frame_count


def determine_nearest(ids: dict, result: dict):
    min_dist = 2000000
    min_id = -1
    for id, value in ids.items():
        if value != None:
            value_cent = (value['center'][0], float(value['center'][1]))
            res_center = (result['center'][0], float(result['center'][1]))
            dist = distance.euclidean(value['center'], result['center'])
            if min_dist > dist:
                min_dist = dist
                min_id = id

    return (min_id, min_dist)


def is_new(min_dist):
    return min_dist > NEW_TRESHOLD

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
ids = {i: None for i in range(100)}
v_data = {i: {} for i in range(100)}
stable_ids = set([])
free_id = 99
duration = get_video_duration(video_cap)[0]
counter = 0
while (True):
    # sleep(5)
    new_ids = {i: None for i in range(100)}
    ret, frame = video_cap.read()
    if ret == True:

        # Write the frame into the file 'output.avi'
        dim = (int(frame_width * 0.5), int(frame_height * 0.5))
        # resize image
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        detections = model.predict(source=frame)[0]
        # initialize the list of bounding boxes and confidences
        results = []
        ######################################
        # DETECTION
        ######################################
        # loop over the detections
        for i, data in enumerate(detections.boxes.data.tolist()):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = data[4]

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            # class_id = int(data[5])
            # add the bounding box (x, y, w, h), confidence and class id to the results list
            center = compute_center_coords(xmin, ymin, xmax, ymax)
            result = {
                            'box': [xmin, ymin, xmax, ymax],
                            'center': center,
                            'confidence': float(confidence)
            }
            #results.append(result)
            len_results_ids = sum([res != None for res in ids.values()])
            #print(len_results_ids)

            if len_results_ids == 0:
                new_ids[i] = result
            else:
                near = determine_nearest(ids, result)
                if (near[1] < 2):
                    new_ids[near[0]] = result

                    ids[near[0]] = None
                else:
                    if is_new(near[1]):
                        new_ids[free_id] = result
                        print('new dist: ', near[1])
                        v_data[free_id][result['center']] = time_ns()
                        free_id -= 1
                    else:
                        # near = determine_nearest(ids, result)
                        #print('Noned: ')
                        new_ids[near[0]] = result
                        v_data[near[0]][result['center']] = time_ns()
                        print('frame: ', counter, ' new id = ', near[0], ': ', near[1])




        for key, value in new_ids.items():
            if value != None:
                xmin, ymin, xmax, ymax = value['box']
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                #cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), GREEN, -1)
                cv2.putText(frame, str(key), (xmin + 5, ymin - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)
                #cv2.putText(frame, str(value['confidence']), (xmin + 10, ymin - 8),
                 #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)


        ids = new_ids
        cv2.imshow('frame', frame)
        cv2.imwrite(f'test_images/frame_{counter}.jpg', frame)
        counter += 1

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

TIME_CONST = duration / ((time_ns() - start) * 1000000000)
X_CONST = 3.1057e-4 / frame_width
Y_CONST = 2.2956e-4 / frame_height


def find_v(data, id):
    data = data[id]
    print(data)
    x = [d[0] for d in data.keys()]
    y = [d[1] for d in data.keys()]
    t = list(data.values())
    current_time = (t[0] - start) * TIME_CONST
    v = []
    print('len:', len(t))
    for i in range(1, len(t)):
        delta_t = (t[i] - t[i - 1]) * TIME_CONST
        # print(delta_t)
        delta_x = x[i] - x[i - 1]
        delta_y = y[i] - y[i - 1]
        v_x = delta_x / delta_t
        v_y = delta_y / delta_t
        v.append((v_x ** 2 + v_y ** 2) ** 0.5)
        # print('time: ', current_time, 'v: ', (v_x ** 2 + v_y ** 2) ** 0.5)

    return {'time': t[1:], 'particle speed': v, 'x': x[1:], 'y': y[1:]}



video_cap.release()
cv2.destroyAllWindows()

pd.DataFrame(find_v(data=v_data, id=98)).to_excel('result.xlsx')
