# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(777, 543)
        MainWindow.setMinimumSize(QtCore.QSize(500, 425))
        MainWindow.setMaximumSize(QtCore.QSize(1000, 756))
        MainWindow.setFocusPolicy(QtCore.Qt.StrongFocus)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.img_lbl = QtGui.QLabel(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_lbl.sizePolicy().hasHeightForWidth())
        self.img_lbl.setSizePolicy(sizePolicy)
        self.img_lbl.setAutoFillBackground(False)
        self.img_lbl.setText(_fromUtf8(""))
        self.img_lbl.setPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/init.png")))
        self.img_lbl.setScaledContents(True)
        self.img_lbl.setObjectName(_fromUtf8("img_lbl"))
        self.verticalLayout_4.addWidget(self.img_lbl)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.lbl_ImgName = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setItalic(False)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.lbl_ImgName.setFont(font)
        self.lbl_ImgName.setFrameShadow(QtGui.QFrame.Plain)
        self.lbl_ImgName.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_ImgName.setObjectName(_fromUtf8("lbl_ImgName"))
        self.verticalLayout_2.addWidget(self.lbl_ImgName)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setSizeConstraint(QtGui.QLayout.SetMinimumSize)
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.btn_Resize = QtGui.QPushButton(self.centralwidget)
        self.btn_Resize.setObjectName(_fromUtf8("btn_Resize"))
        self.horizontalLayout_5.addWidget(self.btn_Resize)
        self.lineEdit_x = QtGui.QLineEdit(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_x.sizePolicy().hasHeightForWidth())
        self.lineEdit_x.setSizePolicy(sizePolicy)
        self.lineEdit_x.setMinimumSize(QtCore.QSize(5, 0))
        self.lineEdit_x.setInputMask(_fromUtf8(""))
        self.lineEdit_x.setText(_fromUtf8(""))
        self.lineEdit_x.setObjectName(_fromUtf8("lineEdit_x"))
        self.horizontalLayout_5.addWidget(self.lineEdit_x)
        self.lineEdit_y = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_y.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_y.sizePolicy().hasHeightForWidth())
        self.lineEdit_y.setSizePolicy(sizePolicy)
        self.lineEdit_y.setMinimumSize(QtCore.QSize(5, 0))
        self.lineEdit_y.setObjectName(_fromUtf8("lineEdit_y"))
        self.horizontalLayout_5.addWidget(self.lineEdit_y)
        self.lineEdit_w = QtGui.QLineEdit(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_w.sizePolicy().hasHeightForWidth())
        self.lineEdit_w.setSizePolicy(sizePolicy)
        self.lineEdit_w.setMinimumSize(QtCore.QSize(5, 0))
        self.lineEdit_w.setObjectName(_fromUtf8("lineEdit_w"))
        self.horizontalLayout_5.addWidget(self.lineEdit_w)
        self.lineEdit_h = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_h.setMinimumSize(QtCore.QSize(5, 0))
        self.lineEdit_h.setObjectName(_fromUtf8("lineEdit_h"))
        self.horizontalLayout_5.addWidget(self.lineEdit_h)
        self.lineEdit_class = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_class.setObjectName(_fromUtf8("lineEdit_class"))
        self.horizontalLayout_5.addWidget(self.lineEdit_class)
        spacerItem = QtGui.QSpacerItem(250, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_5.addItem(spacerItem)
        self.horizontalLayout.addLayout(self.horizontalLayout_5)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.btn_Delete = QtGui.QPushButton(self.centralwidget)
        self.btn_Delete.setObjectName(_fromUtf8("btn_Delete"))
        self.horizontalLayout.addWidget(self.btn_Delete)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout_6.addWidget(self.label)
        self.lineEdit_DeleteLabel = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_DeleteLabel.setObjectName(_fromUtf8("lineEdit_DeleteLabel"))
        self.horizontalLayout_6.addWidget(self.lineEdit_DeleteLabel)
        self.btnDeleteLabels = QtGui.QPushButton(self.centralwidget)
        self.btnDeleteLabels.setObjectName(_fromUtf8("btnDeleteLabels"))
        self.horizontalLayout_6.addWidget(self.btnDeleteLabels)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.btn_SaveIt = QtGui.QPushButton(self.centralwidget)
        self.btn_SaveIt.setObjectName(_fromUtf8("btn_SaveIt"))
        self.verticalLayout_3.addWidget(self.btn_SaveIt)
        self.verticalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 777, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName(_fromUtf8("menuFile"))
        self.menuEdit = QtGui.QMenu(self.menubar)
        self.menuEdit.setObjectName(_fromUtf8("menuEdit"))
        MainWindow.setMenuBar(self.menubar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setMovable(True)
        self.toolBar.setAllowedAreas(QtCore.Qt.AllToolBarAreas)
        self.toolBar.setObjectName(_fromUtf8("toolBar"))
        MainWindow.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBar)
        self.actionLoad = QtGui.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/load.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLoad.setIcon(icon)
        self.actionLoad.setObjectName(_fromUtf8("actionLoad"))
        self.actionEraser = QtGui.QAction(MainWindow)
        self.actionEraser.setEnabled(False)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/eraser.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionEraser.setIcon(icon1)
        self.actionEraser.setObjectName(_fromUtf8("actionEraser"))
        self.actionNext_Image = QtGui.QAction(MainWindow)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/next.jpg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionNext_Image.setIcon(icon2)
        self.actionNext_Image.setObjectName(_fromUtf8("actionNext_Image"))
        self.actionPrevious_Image = QtGui.QAction(MainWindow)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/previous.jpg")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionPrevious_Image.setIcon(icon3)
        self.actionPrevious_Image.setObjectName(_fromUtf8("actionPrevious_Image"))
        self.actionUndo = QtGui.QAction(MainWindow)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(_fromUtf8(":/images/images/undo.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionUndo.setIcon(icon4)
        self.actionUndo.setObjectName(_fromUtf8("actionUndo"))
        self.menuFile.addAction(self.actionLoad)
        self.menuEdit.addAction(self.actionNext_Image)
        self.menuEdit.addAction(self.actionPrevious_Image)
        self.menuEdit.addAction(self.actionUndo)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.toolBar.addAction(self.actionLoad)
        self.toolBar.addAction(self.actionNext_Image)
        self.toolBar.addAction(self.actionPrevious_Image)
        self.toolBar.addAction(self.actionUndo)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Images Labeling Tool", None))
        self.lbl_ImgName.setText(_translate("MainWindow", "Image Name", None))
        self.btn_Resize.setText(_translate("MainWindow", "Resize", None))
        self.lineEdit_x.setPlaceholderText(_translate("MainWindow", "x", None))
        self.lineEdit_y.setPlaceholderText(_translate("MainWindow", "y", None))
        self.lineEdit_w.setPlaceholderText(_translate("MainWindow", "w", None))
        self.lineEdit_h.setPlaceholderText(_translate("MainWindow", "h", None))
        self.lineEdit_class.setPlaceholderText(_translate("MainWindow", "class ", None))
        self.btn_Delete.setText(_translate("MainWindow", "Delete", None))
        self.label.setText(_translate("MainWindow", "Delete labels like", None))
        self.lineEdit_DeleteLabel.setPlaceholderText(_translate("MainWindow", "label to delete from all images", None))
        self.btnDeleteLabels.setText(_translate("MainWindow", "Do it!!!", None))
        self.btn_SaveIt.setText(_translate("MainWindow", "Save all changes", None))
        self.menuFile.setTitle(_translate("MainWindow", "File", None))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit", None))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar", None))
        self.actionLoad.setText(_translate("MainWindow", "Load file...", None))
        self.actionLoad.setShortcut(_translate("MainWindow", "Ctrl+L", None))
        self.actionEraser.setText(_translate("MainWindow", "Eraser", None))
        self.actionEraser.setToolTip(_translate("MainWindow", "Erase events from image", None))
        self.actionNext_Image.setText(_translate("MainWindow", "Next Image", None))
        self.actionNext_Image.setShortcut(_translate("MainWindow", "Right", None))
        self.actionPrevious_Image.setText(_translate("MainWindow", "Previous Image", None))
        self.actionPrevious_Image.setShortcut(_translate("MainWindow", "Left", None))
        self.actionUndo.setText(_translate("MainWindow", "Undo", None))
        self.actionUndo.setShortcut(_translate("MainWindow", "Ctrl+Z", None))

import resources_rc

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

