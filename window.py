import sys

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication

from ui.mainwindow import Ui_BinMainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_BinMainWindow):
    pass


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
