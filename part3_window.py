import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

import source.part3 as part3
from ui.part3_ui import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.execute)

    def execute(self):
        task = self.tasksBox.currentIndex()
        self.output.clear()
        try:
            n1 = int(self.binInput.text().replace('.', ''), 2)
            if task == 0:
                self.output.appendPlainText(part3.poly_to_str(n1))
            elif task == 1:
                n2 = int(self.binInput_2.text().replace('.', ''), 2)
                self.output.appendPlainText(f'{n1} * {n2} = {part3.poly_mul(n1, n2)}')
            elif task == 2:
                self.output.appendPlainText(f'{n1}^-1 = {part3.poly_inv(n1)}')
        except ValueError:
            QMessageBox.information(self, 'Ошибка', 'Не удалось преобразовать число')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
