# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1140, 896)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 30, 1181, 71))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_4.setFont(font)
        self.label_4.setTextFormat(QtCore.Qt.RichText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(720, 200, 381, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
        self.pushButton_5.setSizePolicy(sizePolicy)
        self.pushButton_5.setBaseSize(QtCore.QSize(0, 9))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_2.addWidget(self.pushButton_5)
        self.pushButton_3 = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("color: rgb(255, 255, 255);\n"
"background-color: rgb(255, 0, 0);")
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(720, 460, 371, 41))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_6.sizePolicy().hasHeightForWidth())
        self.pushButton_6.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_4.addWidget(self.pushButton_6)
        self.pushButton_14 = QtWidgets.QPushButton(self.horizontalLayoutWidget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_14.sizePolicy().hasHeightForWidth())
        self.pushButton_14.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_14.setFont(font)
        self.pushButton_14.setObjectName("pushButton_14")
        self.horizontalLayout_4.addWidget(self.pushButton_14)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 150, 661, 691))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_img = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_img.setEnabled(True)
        self.label_img.setAutoFillBackground(False)
        self.label_img.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_img.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_img.setObjectName("label_img")
        self.verticalLayout.addWidget(self.label_img)
        self.label_suturePoint = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_suturePoint.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_suturePoint.setTextFormat(QtCore.Qt.MarkdownText)
        self.label_suturePoint.setObjectName("label_suturePoint")
        self.verticalLayout.addWidget(self.label_suturePoint)
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(720, 260, 381, 41))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.pushButton_2 = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_5.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.horizontalLayoutWidget_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_5.addWidget(self.pushButton)
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(710, 600, 181, 221))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.thedataofrm65 = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.thedataofrm65.setContentsMargins(0, 0, 0, 0)
        self.thedataofrm65.setObjectName("thedataofrm65")
        self.rm65_pz = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_pz.setText("")
        self.rm65_pz.setObjectName("rm65_pz")
        self.thedataofrm65.addWidget(self.rm65_pz, 3, 1, 1, 1)
        self.rm65_ry = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_ry.setText("")
        self.rm65_ry.setObjectName("rm65_ry")
        self.thedataofrm65.addWidget(self.rm65_ry, 5, 1, 1, 1)
        self.rm65_py = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_py.setText("")
        self.rm65_py.setObjectName("rm65_py")
        self.thedataofrm65.addWidget(self.rm65_py, 2, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_11.setEnabled(True)
        self.label_11.setMinimumSize(QtCore.QSize(0, 36))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setLineWidth(6)
        self.label_11.setTextFormat(QtCore.Qt.RichText)
        self.label_11.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.thedataofrm65.addWidget(self.label_11, 0, 0, 1, 2)
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.thedataofrm65.addWidget(self.label_6, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.thedataofrm65.addWidget(self.label_7, 3, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.thedataofrm65.addWidget(self.label_9, 5, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.thedataofrm65.addWidget(self.label_10, 6, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.thedataofrm65.addWidget(self.label_8, 4, 0, 1, 1)
        self.rm65_rz = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_rz.setText("")
        self.rm65_rz.setObjectName("rm65_rz")
        self.thedataofrm65.addWidget(self.rm65_rz, 6, 1, 1, 1)
        self.rm65_rx = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_rx.setText("")
        self.rm65_rx.setObjectName("rm65_rx")
        self.thedataofrm65.addWidget(self.rm65_rx, 4, 1, 1, 1)
        self.rm65_px = QtWidgets.QLabel(self.gridLayoutWidget)
        self.rm65_px.setText("")
        self.rm65_px.setObjectName("rm65_px")
        self.thedataofrm65.addWidget(self.rm65_px, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.thedataofrm65.addWidget(self.label_5, 1, 0, 1, 1)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(920, 600, 191, 221))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.thedataofwound = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.thedataofwound.setContentsMargins(0, 0, 0, 0)
        self.thedataofwound.setObjectName("thedataofwound")
        self.current = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.current.setAlignment(QtCore.Qt.AlignCenter)
        self.current.setObjectName("current")
        self.thedataofwound.addWidget(self.current, 2, 1, 1, 1)
        self.motor_current = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.motor_current.setObjectName("motor_current")
        self.thedataofwound.addWidget(self.motor_current, 2, 2, 1, 1)
        self.current_unit = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.current_unit.setObjectName("current_unit")
        self.thedataofwound.addWidget(self.current_unit, 2, 3, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.thedataofwound.addWidget(self.label_18, 3, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.thedataofwound.addWidget(self.label_16, 4, 3, 1, 1)
        self.speed = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.speed.setAlignment(QtCore.Qt.AlignCenter)
        self.speed.setObjectName("speed")
        self.thedataofwound.addWidget(self.speed, 1, 1, 1, 1)
        self.motor_ifwork = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.motor_ifwork.setObjectName("motor_ifwork")
        self.thedataofwound.addWidget(self.motor_ifwork, 3, 2, 1, 1)
        self.motor_speed_unit = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.motor_speed_unit.setObjectName("motor_speed_unit")
        self.thedataofwound.addWidget(self.motor_speed_unit, 1, 3, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_21.setText("")
        self.label_21.setObjectName("label_21")
        self.thedataofwound.addWidget(self.label_21, 5, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.thedataofwound.addWidget(self.label_3, 5, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_20.setText("")
        self.label_20.setObjectName("label_20")
        self.thedataofwound.addWidget(self.label_20, 5, 2, 1, 1)
        self.motor_speed = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.motor_speed.setObjectName("motor_speed")
        self.thedataofwound.addWidget(self.motor_speed, 1, 2, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.thedataofwound.addWidget(self.label_15, 4, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(0, 36))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setTextFormat(QtCore.Qt.RichText)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.thedataofwound.addWidget(self.label_12, 0, 1, 1, 3)
        self.label_17 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.thedataofwound.addWidget(self.label_17, 3, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.thedataofwound.addWidget(self.label_2, 4, 1, 1, 1)
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(720, 400, 371, 41))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_8 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_8.sizePolicy().hasHeightForWidth())
        self.pushButton_8.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_6.addWidget(self.pushButton_8)
        self.pushButton_10 = QtWidgets.QPushButton(self.horizontalLayoutWidget_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_10.sizePolicy().hasHeightForWidth())
        self.pushButton_10.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.pushButton_10.setFont(font)
        self.pushButton_10.setObjectName("pushButton_10")
        self.horizontalLayout_6.addWidget(self.pushButton_10)
        self.run_control_model = QtWidgets.QLabel(self.centralwidget)
        self.run_control_model.setGeometry(QtCore.QRect(700, 150, 411, 171))
        self.run_control_model.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.run_control_model.setFrameShape(QtWidgets.QFrame.Box)
        self.run_control_model.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.run_control_model.setObjectName("run_control_model")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(700, 350, 411, 221))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(11)
        self.label_22.setFont(font)
        self.label_22.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.label_22.setFrameShape(QtWidgets.QFrame.Box)
        self.label_22.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_22.setObjectName("label_22")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(700, 590, 421, 251))
        self.label.setStyleSheet("background-color: rgb(220, 220, 220);")
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(720, 520, 371, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_11 = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_11.sizePolicy().hasHeightForWidth())
        self.pushButton_11.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_11.setFont(font)
        self.pushButton_11.setObjectName("pushButton_11")
        self.horizontalLayout.addWidget(self.pushButton_11)
        self.label.raise_()
        self.label_22.raise_()
        self.run_control_model.raise_()
        self.label_4.raise_()
        self.horizontalLayoutWidget_2.raise_()
        self.horizontalLayoutWidget_4.raise_()
        self.verticalLayoutWidget.raise_()
        self.horizontalLayoutWidget_5.raise_()
        self.gridLayoutWidget.raise_()
        self.gridLayoutWidget_2.raise_()
        self.horizontalLayoutWidget_6.raise_()
        self.horizontalLayoutWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1140, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionstatue1 = QtWidgets.QAction(MainWindow)
        self.actionstatue1.setObjectName("actionstatue1")
        self.actionstatue2 = QtWidgets.QAction(MainWindow)
        self.actionstatue2.setObjectName("actionstatue2")
        self.actionstatue3 = QtWidgets.QAction(MainWindow)
        self.actionstatue3.setObjectName("actionstatue3")
        self.menu.addSeparator()
        self.menu.addAction(self.actionstatue1)
        self.menu.addAction(self.actionstatue2)
        self.menu.addAction(self.actionstatue3)
        self.menu.addSeparator()
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton_6.clicked.connect(MainWindow.on_testSuture) # type: ignore
        self.pushButton_5.clicked.connect(MainWindow.on_initRM65) # type: ignore
        self.pushButton_2.clicked.connect(MainWindow.on_runRM65) # type: ignore
        self.pushButton_3.clicked.connect(MainWindow.on_stopRM65) # type: ignore
        self.pushButton_8.clicked.connect(MainWindow.on_teach) # type: ignore
        self.pushButton_10.clicked.connect(MainWindow.on_stopTeach) # type: ignore
        self.pushButton_11.clicked.connect(MainWindow.on_imgCollect) # type: ignore
        self.pushButton_14.clicked.connect(MainWindow.on_kinect_pressure) # type: ignore
        self.pushButton.clicked.connect(MainWindow.on_stopCurrentRM) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "自动缝合机器人操作界面"))
        self.pushButton_5.setText(_translate("MainWindow", "初始化"))
        self.pushButton_3.setText(_translate("MainWindow", "急停"))
        self.pushButton_6.setText(_translate("MainWindow", "缝合下针"))
        self.pushButton_14.setText(_translate("MainWindow", "位姿记录"))
        self.label_img.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label_suturePoint.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.pushButton_2.setText(_translate("MainWindow", "开始缝合"))
        self.pushButton.setText(_translate("MainWindow", "换针"))
        self.label_11.setText(_translate("MainWindow", "机械臂末端位姿"))
        self.label_6.setText(_translate("MainWindow", "Py(mm)"))
        self.label_7.setText(_translate("MainWindow", "Pz(mm)"))
        self.label_9.setText(_translate("MainWindow", "Ry(rad)"))
        self.label_10.setText(_translate("MainWindow", "Rz(rad)"))
        self.label_8.setText(_translate("MainWindow", "Rx(rad)"))
        self.label_5.setText(_translate("MainWindow", "Px(mm)"))
        self.current.setText(_translate("MainWindow", "电流"))
        self.motor_current.setText(_translate("MainWindow", "30"))
        self.current_unit.setText(_translate("MainWindow", "mA"))
        self.label_18.setText(_translate("MainWindow", "是否正常"))
        self.speed.setText(_translate("MainWindow", "转速"))
        self.motor_ifwork.setText(_translate("MainWindow", "是"))
        self.motor_speed_unit.setText(_translate("MainWindow", "rad/s"))
        self.motor_speed.setText(_translate("MainWindow", "2"))
        self.label_12.setText(_translate("MainWindow", "缝合机构参数"))
        self.pushButton_8.setText(_translate("MainWindow", "示教开始"))
        self.pushButton_10.setText(_translate("MainWindow", "示教停止"))
        self.run_control_model.setText(_translate("MainWindow", "<html><head/><body><p align=\"justify\"><span style=\" font-size:14pt;\">运动模块</span></p></body></html>"))
        self.label_22.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt;\">示教模块</span></p></body></html>"))
        self.pushButton_11.setText(_translate("MainWindow", "示教前数据采集"))
        self.menu.setTitle(_translate("MainWindow", "缝合模式"))
        self.actionstatue1.setText(_translate("MainWindow", "statue1"))
        self.actionstatue2.setText(_translate("MainWindow", "statue2"))
        self.actionstatue3.setText(_translate("MainWindow", "statue3"))
