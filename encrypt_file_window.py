import os
import sys
from copy import copy
from threading import Thread
from typing import List, cast

from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QFileDialog, QMessageBox, QPushButton

from source.algorithms import AlgorithmConnect, BaseAlgorithm, CryptoMode
from source.data import ENCRYPTED_FILE_EXTENSION
from source.signals import UpdateSignal
from ui.crypto_file_widget import Ui_CryptoWidget
from ui.crypto_window import Ui_CryptoWindow
from ui.open_file_choice import Ui_ChoiceDialog


class MainWindow(QMainWindow, Ui_CryptoWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowIcon(QIcon('./icons8-protect-64.png'))
        self.visible_icon = QIcon('./ui/icons8-eye-24.png')
        self.invisible_icon = QIcon('./ui/icons8-invisible-24.png')
        self.passShowButton.setIcon(self.invisible_icon)
        self.passShowButton.clicked.connect(self.vis_invis_change)
        for item in AlgorithmConnect.reverse_connect:
            self.algoBox.addItem(item)
        for item in AlgorithmConnect.mode_connect:
            self.modeBox.addItem(item)
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
        def thread_part(thread_widget: Ui_CryptoWidget, s_state: bool, s_signal):
            file_name = thread_widget.originalFile if s_state else thread_widget.encryptedFile
            thread_widget.progressBar.setVisible(True)
            extension = thread_widget.originalFile.rsplit('.', 1)[-1]
            mode = AlgorithmConnect.mode_connect[self.modeBox.currentText()]
            alg = AlgorithmConnect.reverse_connect[self.algoBox.currentText()]

            file = BaseAlgorithm.encrypt_decrypt(file_name, mode, alg, s_signal, self.passEdit.text(),
                                                 decryption=not s_state, ext=extension)
            thread_widget.progressBar.setVisible(False)

            if file is None:
                if s_state:
                    QMessageBox.information(self, 'Ошибка', 'Не удалось найти входной файл')
                    thread_widget.workButton.setChecked(not s_state)
                    thread_widget.workButton.setText('Зашифровать' if s_state else 'Дешифровать')
                else:
                    QMessageBox.information(self, 'Ошибка', 'Не удалось найти зашифрованный файл\n'
                                                            'Файл будет зашифрован повторно')
                    thread_widget.progressBar.setVisible(True)
                    file = BaseAlgorithm.encrypt_decrypt(thread_widget.originalFile, mode, alg, s_signal,
                                                         self.passEdit.text(), decryption=False, ext=extension)
                    thread_widget.progressBar.setVisible(False)
                    if file is None:
                        QMessageBox.critical(self, 'Ошибка', 'Похоже файл потерян окончательно'
                                                             'Файл будет удален из списка')
                        self.remove_crypto_widget(file_name)
                    else:
                        thread_widget.encryptedFile = file
            else:
                thread_widget.workButton.setText('Дешифровать' if s_state else 'Зашифровать')
                if s_state:
                    thread_widget.encryptedFile = file
                else:
                    thread_widget.decryptedFile = file

        key = self.passEdit.text()
        if key == '':
            QMessageBox.information(self, 'Нет ключа', 'Ключ не может быть пустым')
            button = self.sender().parent().workButton
            button = cast(QPushButton, button)
            button.setChecked(not state)
            button.setText('Зашифровать')
            return
        widget = self.sender().parent()
        widget = cast(Ui_CryptoWidget, widget)
        signal = UpdateSignal()
        signal.update.connect(widget.progressBar.setValue)
        thread = Thread(target=thread_part, args=(widget, state, signal), daemon=True)
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
