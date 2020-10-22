from PyQt4 import QtGui

import cv2


def is_in(list_names, search_list):
    for f in list_names:
        if f not in search_list:
            return False
    return True


def show_message(message, title="Error!"):
    e = QtGui.QMessageBox()
    e.setWindowTitle(title)
    e.setText(message)
    e.exec_()


def draw_bBox(img, start, end, class_name, prob=None, bbox_color=(0, 255, 0)):
    """
    draw bounding box in image.
    start is a tuple of (x, y) top left pixel
    end is a tuple of bottom right pixel
    """
    cv2.rectangle(img, start, end, bbox_color, 1)
    if prob:
	prob = int(prob)

    if class_name and class_name != "":
        text = "{0}:{1}%".format(class_name, prob) if prob else class_name
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        # print((start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]))
        cv2.rectangle(img, (start[0], start[1]-text_size[1]), (start[0]+text_size[0], start[1]), bbox_color, cv2.FILLED)
        cv2.putText(img, text, start, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)


def get_bboxes(img_data):
    bboxes = []
    for _, r in img_data.iterrows():
        bboxes.append((r.x, r.y, r.w, r.h, r.class_name, r.proba))
    return bboxes


def remove_extension(s):
    if "." not in s:
        return s
    r = s.split(".")
    return ".".join(r[:-1])
