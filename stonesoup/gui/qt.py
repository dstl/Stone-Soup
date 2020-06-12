import sys

if 'PySide2' in sys.modules:
    # PySide2
    from PySide2 import QtGui, QtWidgets, QtCore, Qt
    from PySide2.QtCore import Signal, Slot
else:
    # PyQt5
    from PyQt5 import QtGui, QtWidgets, QtCore, Qt
    from PyQt5.QtCore import pyqtSignal as Signal, pyqtSlot as Slot
