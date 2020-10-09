import sys
import source.Part1 as part1
from source.exceptions import ParamException

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from ui.binmainwindow import Ui_BinMainWindow


class MainWindow(QMainWindow, Ui_BinMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.tasks = [part1.task1, part1.task2, part1.task3, part1.task4,
                      part1.task5, part1.task6, part1.task7, part1.task8]

        self.pushButton.clicked.connect(self.task_execute)
        self.input.textChanged.connect(self.input_changed)
        self.updateButton.clicked.connect(self.binInput_changed)
        # self.binInput.textEdited.connect(self.binInput_changed)

    def binInput_changed(self):
        # self.input.clear()
        self.input.setText(str(int(self.binInput.text().replace('.', ''), 2)))

    def input_changed(self, text: str):
        try:
            num = int(text)
        except ValueError:
            num = 0
        self.binInput.setText(bin(num)[2:])

    def task_execute(self):
        self.output.clear()
        try:
            res = self.tasks[self.tasksBox.currentIndex()](
                int(self.binInput.text().replace('.', ''), 2), self.lineEdit.text()
            )
        except ParamException as e:
            QMessageBox.critical(self.centralwidget, 'Ошибка', str(e))
            return
        except Exception as e:
            print(e)
            return
        self.output.setPlainText('\n'.join(res))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
