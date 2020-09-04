import numbers
from typing import Any, Sequence

import numpy as np
from .qt import QtCore, Qt, QtWidgets, Signal

import stonesoup
from stonesoup.gui.gui_logging import LOG, LogFunction
from stonesoup.types.array import StateVector


class StoneSoupModel(QtCore.QObject):
    dataChanged = Signal()

    def __init__(self, parent: QtCore.QObject, obj: stonesoup.types.base.Base):
        super().__init__(parent)
        self.obj = obj

        def setattr_with_signal(original_object, key, value):
            LOG.debug('set attr with signal called')
            super(original_object.__class__, original_object).__setattr__(key, value)
            # noinspection PyUnresolvedReferences
            self.dataChanged.emit()
        self.obj.__class__.__setattr__ = setattr_with_signal

    def __getattr__(self, item):
        if item in ('obj', 'dataChanged'):
            return super(QtCore.QObject, self).__getattribute__(item)
        else:
            return getattr(self.obj, item)

    def __setattr__(self, key, value):
        LOG.debug('set attr of model called')
        if key in ('obj', 'dataChanged'):
            super(QtCore.QObject, self).__setattr__(key, value)
        else:
            self.obj.__setattr__(key, value)


class OneDNumericModel(QtCore.QAbstractListModel):
    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = None
        self.resetData(new_data=data)

    def flags(self, index: QtCore.QModelIndex) -> Qt.Qt.ItemFlags:
        return Qt.Qt.ItemFlags(Qt.Qt.ItemIsSelectable | Qt.Qt.ItemIsEditable | Qt.Qt.ItemIsEnabled)

    def data(self, index: QtCore.QModelIndex, role: int = Qt.Qt.DisplayRole) -> Any:
        # print(f'Getting _data index: {index}, role: {role} (Display is {Qt.Qt.DisplayRole})')
        if role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            # See below for the _data structure.
            return str(self._data[index.row()])

    def setData(self, index: QtCore.QModelIndex, value: Any,
                role: int = Qt.Qt.DisplayRole) -> bool:
        assert index.column() == 0
        self._data[index.row()] = float(value)
        self.dataChanged.emit(index, index)
        return True

    def rowCount(self, parent: QtCore.QModelIndex = None, *args, **kwargs) -> int:
        length = len(self._data)
        # print(length)
        return length

    # noinspection PyPep8Naming
    def resetData(self, new_data: Sequence[numbers.Number]) -> None:
        if new_data is None:
            new_data = StateVector([])
        if isinstance(new_data, tuple):
            new_data = list(new_data)
        self._data = new_data
        top_left = self.createIndex(0, 0)
        bottom_right = self.createIndex(self.rowCount(), 0)
        self.dataChanged.emit(top_left, bottom_right)


# noinspection PyCompatibility
class NDArrayModel(QtCore.QAbstractTableModel):
    @LogFunction()
    def __init__(self, *args, array: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)
        LOG.debug(f'NDArrayModel.__init__ array = {array}')
        self.array = None
        self.resetData(array)

    # noinspection PyPep8Naming
    def resetData(self, array: np.ndarray) -> None:
        if array is None:
            array = np.zeros((0, 0))
        self.array = np.asarray(array)
        LOG.debug(f'NDArrayModel.__init__ self.array = {repr(self.array)}')
        if not self.array.ndim <= 2:
            raise ValueError('View not implemented for arrays with more than 2 dimensions')
        top_left = self.createIndex(0, 0)
        bottom_right = self.createIndex(self.rowCount(), self.columnCount())
        self.dataChanged.emit(top_left, bottom_right)

    @LogFunction()
    def data(self, index: QtCore.QModelIndex, role: int) -> Any:
        LOG.debug(f'NDArrayModel.data index = ({index.row()}, {index.column()})')
        LOG.debug(f'NDArrayModel.data ndim = {self.array.ndim}')
        LOG.debug(f'NDArrayModel.data role = {role} (Display is {Qt.Qt.DisplayRole})')
        if role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            if self.array.ndim == 2:
                value = self.array[index.row(), index.column()]
            elif self.array.ndim == 1:
                value = self.array[index.column()]
            else:
                raise NotImplementedError
            LOG.debug(f'NDArrayModel.data value = {value}')
            return str(value)

    @LogFunction()
    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        LOG.debug(f'NDArrayModel.rowCount ndim = {self.array.ndim}')
        if self.array.ndim == 2:
            rows = self.array.shape[1]
        elif self.array.ndim == 1:
            rows = self.array.shape[0]
        else:
            raise NotImplementedError
        return rows

    @LogFunction()
    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if self.array.ndim == 2:
            cols = self.array.shape[0]
        elif self.array.ndim == 1:
            cols = 1
        else:
            raise NotImplementedError
        return cols


class SpinBoxDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent: QtWidgets.QWidget, option: QtWidgets.QStyleOptionViewItem,
                     index: QtCore.QModelIndex):
        editor = QtWidgets.QSpinBox(parent)
        editor.setFrame(False)
        return editor

    def setEditorData(self, editor: QtWidgets.QWidget, index: QtCore.QModelIndex):
        value = int(index.model().data(index, Qt.Qt.EditRole))
        editor.setValue(value)

    def setModelData(self, editor: QtWidgets.QWidget, model: QtCore.QAbstractItemModel,
                     index: QtCore.QModelIndex):
        # QSpinBox *spinBox = static_cast<QSpinBox*>(editor);
        editor.interpretText()
        value = editor.value()
        model.setData(index, value, Qt.Qt.EditRole)

    def updateEditorGeometry(self, editor: QtWidgets.QWidget,
                             option: QtWidgets.QStyleOptionViewItem,
                             index: QtCore.QModelIndex) -> None:
        editor.setGeometry(option.rect)
