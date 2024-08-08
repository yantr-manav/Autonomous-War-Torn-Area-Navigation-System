from ultralytics import YOLO
import cv2
import math

def detect():
    # start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 900)
    cap.set(4, 1000)

    # model
    model = YOLO("best.pt")

    # object classes

    classNames = [
        "human_aid_rehabilitation",
        "fire",
        "military_vehicles",
        "destroyed_buildings",
        "combat"
    ]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes
            keys_arr = ["A", "B", "C", "D", "E"]
            coor_arr = []
            val_arr = []

            # To get the coordinates
            for box in boxes:
                x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                coor_arr.append([x, y])

            # To sort the coordinates in descending order (bottom to top)
            sorted_coor_array = sorted(
                coor_arr, key=lambda point: point[1], reverse=True)

            # To get the cls values by equating the coor
            for arr in sorted_coor_array:
                for box in boxes:
                    box_arr = [int(box.xyxy[0][0]), int(box.xyxy[0][1])]
                    if arr == box_arr:
                        cls_num = int(box.cls[0])
                        name = classNames[cls_num]
                        val_arr.append(name)

            # FINAL LABEL
            identified_labels = dict(zip(keys_arr, val_arr))
            print(identified_labels)

            # For Bounding Boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(
                    x2), int(y2)  # convert to int values

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (0, 255, 0)
                thickness = 2

                cls = int(box.cls[0])
                x, y = int(box.xyxy[0][0]), int(box.xyxy[0][1])
                clsPlusConf = classNames[cls] + str(confidence)
                cv2.putText(img, clsPlusConf, org,
                            font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return identified_labels

##############################################################


def task_5b_return():
    """
    Purpose:
    ---
    Only for returning the final dictionary variable

    Arguments:
    ---
    You are not allowed to define any input arguments for this function. You can 
    return the dictionary from a user-defined function and just call the 
    function here

    Returns:
    ---
    `identified_labels` : { dictionary }
        dictionary containing the labels of the events detected
    """
    identified_labels = {}

############## ADD YOUR CODE HERE	##############
    identified_labels = detect()

##################################################
    return identified_labels


############### Main Function	#################
if __name__ == "__main__":
    identified_labels = task_5b_return()
    print(identified_labels)

