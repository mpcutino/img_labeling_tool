from os import path as os_path
from os import remove
from PyQt4 import QtGui, QtCore

import pandas as pd
import numpy as np
import cv2
import time

from qt_generated.main import Ui_MainWindow, _fromUtf8
from utils import is_in, show_message, draw_bBox, get_bboxes, remove_extension

REQUIRED_COLS = ["timestamp", "boxImg_name", "class_name", "proba", "x", "y", "w", "h"]
IMG_W = 346
IMG_H = 260


class Modified_MainWindow(Ui_MainWindow):

    def __init__(self):
        super(Modified_MainWindow, self).__init__()

        self.images_df = pd.DataFrame()
        self.images_names = []
        self.count = -1
        self.file_path = ""
        self.img_folder = ""
        self.box_def_color = (0, 255, 0)
        self.box_attention_color = (255, 0, 0)
        self.box_selected_color = (0, 0, 255)
        self.attention_box = ()
        self.modified_boxes = []
        self.last_clicked = ()
        self.start_click = 0

    def doUiSetup(self, qtMainWindow):
        # this is from base class
        self.setupUi(qtMainWindow)

        # now add my own stuff
        self.mainWindow = qtMainWindow
        # add connections to actions
        self.actionLoad.triggered.connect(self.load_file)
        self.actionNext_Image.triggered.connect(self.next_image)
        self.actionPrevious_Image.triggered.connect(self.prev_image)
        self.actionUndo.triggered.connect(self.undo)
        # add event to image click
        self.img_lbl.mousePressEvent = self.img_clicked
        self.img_lbl.mouseReleaseEvent = self.mouse_released
        # add connection to buttons
        self.btn_Delete.clicked.connect(self.remove_att_box)
        self.btn_Resize.clicked.connect(self.resize_box)
        self.btn_SaveIt.clicked.connect(self.save_df)
        self.btnDeleteLabels.clicked.connect(self.delete_labels)
        # connect to exit event
        self.mainWindow.closeEvent = self.close_event

    def delete_labels(self):
        label = str(self.lineEdit_DeleteLabel.text())
        if len(label):
            self.images_df = self.images_df.query("class_name != @label").reset_index(drop=True)
            self.lineEdit_DeleteLabel.setText("")
            self.lineEdit_DeleteLabel.setPlaceholderText("Done!!")
            # update image. setting count in -1 to simulate that we are in the previous image
            self.count -= 1
            self.update_image(foward=True)

    def mouse_released(self, event):
        end = time.time()
        x = event.pos().x()
        y = event.pos().y()
        # transform from label scale to image scale
        x = int(x * float(IMG_W) / self.img_lbl.width())
        y = int(y * float(IMG_H) / self.img_lbl.height())

        current_img_name = self.images_names[self.count]
        img_data = self.images_df.loc[self.images_df["boxImg_name"] == current_img_name]
        box_x, box_y, w, h, cl_name = '', '', '', '', ''
        boxes = []
        self.attention_box = ()

        if len(self.last_clicked) and (end - self.start_click) > 0.2:
            origin_x, origin_y = self.last_clicked
            x_min, x_max = min(x, origin_x), max(x, origin_x)
            y_min, y_max = min(y, origin_y), max(y, origin_y)
            # x, y, w, h, cl_name, prob, bbox_color
            box_x, box_y, w, h, cl_name = (x_min, y_min, (x_max-x_min), (y_max-y_min), '')
            boxes.append((box_x, box_y, w, h, cl_name, '', self.box_selected_color))
            boxes.extend(get_bboxes(img_data))
        else:
            for img_box_x, img_box_y, img_w, img_h, img_cl_name, img_prob in get_bboxes(img_data):
                color = self.box_def_color
                if img_box_x <= x <= (img_box_x + img_w) and img_box_y <= y <= (img_box_y + img_h):
                    color = self.box_attention_color
                    self.attention_box = (img_box_x, img_box_y, img_w, img_h, img_cl_name, img_prob)
                boxes.append((img_box_x, img_box_y, img_w, img_h, img_cl_name, img_prob, color))
        if len(self.attention_box):
            box_x, box_y, w, h, cl_name, _ = self.attention_box
        self.lineEdit_x.setText(str(box_x))
        self.lineEdit_y.setText(str(box_y))
        self.lineEdit_w.setText(str(w))
        self.lineEdit_h.setText(str(h))
        self.lineEdit_class.setText(str(cl_name))
        self.paint_img(boxes)

    def resize_box(self):
        try:
            x = int(self.lineEdit_x.text())
            y = int(self.lineEdit_y.text())
            w = int(self.lineEdit_w.text())
            h = int(self.lineEdit_h.text())
            cl_name = str(self.lineEdit_class.text())
            # chequear si son iguales. En ese caso no hay que hacer nada, para no duplicar
            if len(self.attention_box) and self.attention_box[:5] == (x, y, w, h, cl_name):
                return
            # en caso contrario, actualizar y, si habia un box en attention modificar su valor

            if (not self.__is_valid(x, y)) or w < 5 or h < 5:
                show_message("Variables must be integers bigger than 0. Height and width must be bigger than 5")
            elif not len(cl_name):
                show_message("Class name must not be empty")
            else:
                # REQUIRED_COLS = ["timestamp", "boxImg_name", "class_name", "proba", "x", "y", "w", "h"]
                b_name = str(self.lbl_ImgName.text())
                self.images_df.loc[len(self.images_df.index)] = [remove_extension(b_name),
                                                                 b_name, cl_name, 1, x, y, w, h, "manual"]
                print(len(self.images_df.index))
                if len(self.attention_box):
                    self.remove_att_box()
                else:
                    self.paint_img(get_bboxes(self.images_df.loc[self.images_df["boxImg_name"] == b_name]))
        except Exception as e:
            print e
            show_message("Variables must be integers bigger than 0. Height and width must be bigger than 5")

    def remove_att_box(self):
        if 0 <= self.count < len(self.images_names) and len(self.attention_box):
            x, y, w, h, cl_name, prob = self.attention_box
            current_img_name = self.images_names[self.count]
            df = self.images_df.query("boxImg_name == @current_img_name & "
                                                  "class_name == @cl_name & proba == @prob & "
                                                  "x == @x & y == @y & w == @w & h == @h")
            self.images_df = self.images_df.query("boxImg_name != @current_img_name | "
                                                  "class_name != @cl_name | proba != @prob | "
                                                  "x != @x | y != @y | w != @w | h != @h").reset_index(drop=True)
            img_data = self.images_df.loc[self.images_df["boxImg_name"] == current_img_name]
            self.paint_img(get_bboxes(img_data))
            self.modified_boxes.append(df)
            self.attention_box = ()

    def close_event(self, event):
        reply = QtGui.QMessageBox.question(self.mainWindow, 'Message',
                                           "Are you sure to quit? Be sure to save changes!!", QtGui.QMessageBox.Yes,
                                           QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def save_df(self):
        # guardar el dataframe con los cambios
        if len(self.file_path):
            folder = os_path.dirname(self.file_path)
            self.images_df.sort_values(by=['boxImg_name']).to_csv(os_path.join(folder, "edited_annotate.txt"), index=False)

    def load_file(self):
        # The QWidget widget is the base class of all user interface objects in PyQt4.
        w = QtGui.QWidget()
        # Set window size.
        w.resize(320, 240)
        # Set window title
        w.setWindowTitle("Hello World!")
        filename = QtGui.QFileDialog.getOpenFileName(w, 'Open File', '/', "Text Files (*.txt *.csv)")
        print(filename)

        try:
            df = pd.read_csv(str(filename))
            if not is_in(REQUIRED_COLS, df.columns):
                # display error message
                show_message("Dataframe must contain columns:\n\n{0}".format(", ".join(REQUIRED_COLS)))
            else:
                self.file_path = str(filename)
                folder = os_path.dirname(self.file_path)
                self.images_df = df
                self.img_folder = os_path.join(folder, "clean_images")
                self.images_names = list(df.boxImg_name.unique())
                self.count = -1
                self.update_image(foward=True)
        except Exception as e:
            print(e)
            show_message("Not a valid dataframe")

    def img_clicked(self, event):
        if self.count < 0 or self.count >= len(self.images_names):
            return
        x = event.pos().x()
        y = event.pos().y()
        # transform from label scale to image scale
        x = int(x * float(IMG_W) / self.img_lbl.width())
        y = int(y * float(IMG_H) / self.img_lbl.height())
        self.last_clicked = (x, y)
        self.start_click = time.time()

    def undo(self):
        if len(self.modified_boxes):
            df = self.modified_boxes[-1]
            self.images_df = self.images_df.append(df).reset_index(drop=True)
            self.paint_img(get_bboxes(df))
            self.modified_boxes = self.modified_boxes[:-1]

    def next_image(self):
        self.update_image(foward=True)

    def prev_image(self):
        self.update_image(foward=False)

    def update_image(self, foward=True):
        # new image, so update clicked pixels
        self.__init_clck_pixels()
        if not len(self.images_names):
            show_message("No images")
            return

        self.count += 1 if foward else -1

        if self.count >= len(self.images_names) and foward:
            self.count = 0
        if self.count < 0 and not foward:
            self.count = len(self.images_names) - 1

        current_img_name = self.images_names[self.count]
        img_data = self.images_df.loc[self.images_df["boxImg_name"] == current_img_name]
        self.paint_img(get_bboxes(img_data))
        self.lbl_ImgName.setText(self.images_names[self.count])

    def paint_img(self, bboxes):
        if 0 <= self.count < len(self.images_names):
            current_img_name = self.images_names[self.count]
            img_path = os_path.join(self.img_folder, current_img_name)
            img = cv2.imread(img_path)
            for box in bboxes:
                if len(box) == 6:
                    x, y, w, h, cl_name, prob = box
                    bbox_color = self.box_def_color
                else:
                    x, y, w, h, cl_name, prob, bbox_color = box
                draw_bBox(img, (x, y), (x+w, y+h), cl_name, prob, bbox_color=bbox_color)
            cv2.imwrite("tmp.png", img)
            self.img_lbl.setPixmap(QtGui.QPixmap("tmp.png"))
            remove("tmp.png")
            # self.img_lbl.setPixmap(QtGui.QPixmap(img_path))

    def __init_clck_pixels(self):
        self.modified_boxes = []
        self.attention_box = ()
        self.last_clicked = ()
        self.lineEdit_x.setText("")
        self.lineEdit_y.setText("")
        self.lineEdit_w.setText("")
        self.lineEdit_h.setText("")
        self.lineEdit_class.setText("")

    @staticmethod
    def __is_valid(x, y):
        return 0 <= x < IMG_W and 0 <= y < IMG_H


if __name__ == "__main__":
    import sys

    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Modified_MainWindow()
    ui.doUiSetup(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
