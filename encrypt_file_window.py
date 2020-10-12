import os
import sys
from copy import copy
from threading import Thread
from typing import List, cast

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QDialog, QFileDialog, QMessageBox, QPushButton

from source.data import ENCRYPTED_FILE_EXTENSION
from ui.crypto_window import Ui_CryptoWindow
from ui.crypto_file_widget import Ui_CryptoWidget
from ui.open_file_choice import Ui_ChoiceDialog
from source.crypto_algorithm import Algorithm


class MainWindow(QMainWindow, Ui_CryptoWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.visible_icon = QIcon('./ui/icons8-eye-24.png')
        self.invisible_icon = QIcon('./ui/icons8-invisible-24.png')
        self.passShowButton.setIcon(self.invisible_icon)
        self.passShowButton.clicked.connect(self.vis_invis_change)
        self.algoBox.addItem(Algorithm.vernam_cipher.__doc__)
        self.addFile.triggered.connect(self.add_file)
        self.crypto_widget_list = {}

    def vis_invis_change(self, checked: bool):
        self.passShowButton.setIcon(self.visible_icon if checked else self.invisible_icon)
        self.passEdit.setEchoMode(QLineEdit.Normal if checked else QLineEdit.Password)

    def add_file(self):
        files, _ = QFileDialog.getOpenFileNames(self, 'Выбирите файл или файлы', os.curdir)
        files = cast(List, files)
        for file in copy(files):
            if os.path.exists(file):
                self.add_cryptofile(file)
                files.remove(file)

    def add_cryptofile(self, file: str):
        if file in self.crypto_widget_list:
            QMessageBox.information(self, 'Сообщение', 'Файл уже добавлен')
            return
        widget = Ui_CryptoWidget(file)
        widget.setupUi(widget)
        widget.progressBar.setVisible(False)
        self.widgetSpace.addWidget(widget)
        self.crypto_widget_list[file] = widget

        widget.fileButton.setText(file.split('/')[-1].split('\\')[-1])
        widget.workButton.setChecked(file.split('.')[-1] == ENCRYPTED_FILE_EXTENSION)

        widget.fileButton.clicked.connect(self.open_file)
        widget.workButton.clicked.connect(self.en_decrypt_file)

    def en_decrypt_file(self, state: bool):  # state = True -- encrypt
        def thread_part(widget: Ui_CryptoWidget, s_state: bool):
            # widget = cast(Ui_CryptoWidget, widget)
            try:
                file_name = widget.originalFile if s_state else widget.encryptedFile
                widget.progressBar.setVisible(True)
                file = Algorithm.reverse_connect[self.current_algo](file_name, key, widget.progressBar.setValue,
                                                                    decryption=not s_state)
                widget.progressBar.setVisible(False)
                if file is None:
                    if s_state:
                        QMessageBox.information(self, 'Ошибка', 'Не удалось найти входной файл')
                        widget.workButton.setChecked(not s_state)
                        widget.workButton.setText('Зашифровать' if s_state else 'Дешифровать')
                    else:
                        QMessageBox.information(self, 'Ошибка', 'Не удалось найти зашифрованный файл\n'
                                                                'Файл будет зашифрован повторно')
                        file = Algorithm.reverse_connect[self.current_algo](file_name, key, widget.progressBar.setValue,
                                                                            decryption=not s_state)
                        if file is None:
                            QMessageBox.critical(self, 'Ошибка', 'Похоже файл потерян окончательно'
                                                                 'Файл будет удален из списка')
                            self.remove_crypto_widget(file_name)
                        else:
                            widget.encryptedFile = file
                else:
                    widget.workButton.setText('Дешифровать' if s_state else 'Зашифровать')
                    if s_state:
                        widget.encryptedFile = file
                    else:
                        widget.decryptedFile = file
            except Exception as e:
                print(e)

        key = self.passEdit.text()
        if key == '':
            QMessageBox.information(self, 'Нет ключа', 'Ключ не может быть пустым')
            button = self.sender().parent().workButton
            button = cast(QPushButton, button)
            button.setChecked(not state)
            button.setText('Зашифровать')
            return
        widget = self.sender().parent()
        thread = Thread(target=thread_part, args=(widget, state), daemon=True)
        thread.start()

    @property
    def current_algo(self):
        return self.algoBox.currentData(Qt.DisplayRole)

    def open_file(self):
        widget = self.sender().parent()
        widget = cast(Ui_CryptoWidget, widget)
        dialog = Ui_ChoiceDialog()
        if dialog.exec_():
            if dialog.choice == 0:
                if widget.originalFile is None:
                    self.file_dont_exist()
                else:
                    os.system(f'"{widget.originalFile}"')
            elif dialog.choice == 1:
                if widget.encryptedFile is None:
                    self.file_dont_exist()
                else:
                    os.system(f'"{widget.encryptedFile}"')
            elif dialog.choice == 2:
                if widget.decryptedFile is None:
                    self.file_dont_exist()
                else:
                    os.system(f'"{widget.decryptedFile}"')

    def file_dont_exist(self):
        QMessageBox.information(self, 'Сообщение', 'Невозможно открыть файл, так как он еще не был создан')

    def remove_crypto_widget(self, file: str):
        self.widgetSpace.removeWidget(self.crypto_widget_list[file])
        self.crypto_widget_list.pop(file)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()
