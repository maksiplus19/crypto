import sys
from collections import Iterable

from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

import source.part2 as part2
from ui.second_part_window import Ui_SecondPartWindow


class MainWindow(QMainWindow, Ui_SecondPartWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.combo_box.addItems([
            '1. Простые числа (a)',
            '2. Приведенная система вычетов (a)',
            '3. Функция Эйлера (a)',
            '4. Разложение на простые числа (a)',
            '5. Возведение в степень по модулю (a degree mod)',
            '6. Алгоритм Евклида (a b)',
            '7. Бинарный алгоритм Евклида (a b)',
        ])
        self.execute_button.clicked.connect(self.execute)

    def execute(self):
        task = self.combo_box.currentIndex()
        self.plain_text_edit.clear()
        text = self.line_edit.text()
        if ' ' in text:
            try:
                text = text.split(' ')
                m = list(map(int, text))
            except ValueError:
                QMessageBox.critical(self, 'Ошибка', 'Не удалось преобразовать в число')
                return
        else:
            try:
                m = int(text)
            except ValueError:
                QMessageBox.critical(self, 'Ошибка', 'Не удалось преобразовать в число')
                return
        if task == 0:
            if isinstance(m, Iterable):
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            gen = part2.prime_gen()
            while (i := next(gen)) < m:
                self.plain_text_edit.appendPlainText(f'{i} ')
        elif task == 1:
            if isinstance(m, Iterable):
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            [self.plain_text_edit.appendPlainText(i) for i in map(str, part2.prime_system(m))]
        elif task == 2:
            if isinstance(m, Iterable):
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            self.plain_text_edit.appendPlainText(f'phi({m}) = {part2.phi(m)}')
        elif task == 3:
            if isinstance(m, Iterable):
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            [self.plain_text_edit.appendPlainText(i) for i in map(str, part2.prime_decomposition(m))]
        elif task == 4:
            if isinstance(m, int) or len(m) != 3:
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            self.plain_text_edit.appendPlainText(f'{m[0]}^{m[1]} mod {m[2]} = '
                                                 f'{part2.binary_power_mod(m[0], m[1], m[2])}')
        elif task == 5:
            if isinstance(m, int) or len(m) != 2:
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            res = part2.gcd_ext(m[0], m[1])
            self.plain_text_edit.appendPlainText(f'gcd({m[0], m[1]}) = {res[0]} = {res[1]}*{m[0]} + {res[2]}*{m[1]}')
        elif task == 6:
            if isinstance(m, int) or len(m) != 2:
                QMessageBox.critical(self, 'Ошибка', 'Некорректный ввод')
                return
            self.plain_text_edit.appendPlainText(f'{part2.gcd_binary(m[0], m[1])}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
