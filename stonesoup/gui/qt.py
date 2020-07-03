import sys
# flake8: noqa F401

if 'PySide2' in sys.modules:
    # PySide2
    # noinspection PyUnresolvedReferences
    from PySide2 import QtGui, QtWidgets, QtCore, Qt
    # noinspection PyUnresolvedReferences
    from PySide2.QtCore import Signal, Slot
else:
    # PyQt5
    # noinspection PyUnresolvedReferences
    from PyQt5 import QtGui, QtWidgets, QtCore, Qt
    # noinspection PyUnresolvedReferences
    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
