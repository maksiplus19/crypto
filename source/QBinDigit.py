from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPainter, QMouseEvent, QFont
from PyQt5.QtWidgets import QWidget, QSizePolicy, QLCDNumber
from PyQt5.QtGui import QPaintEvent
from PyQt5.Qt import Qt


class QBinDigit(QLCDNumber):
    def __init__(self, centralWidget):
        super().__init__(centralWidget)


    def mousePressEvent(self, event: QMouseEvent) -> None:
        print(event.x(), event.y())

