import decimal
import functools
import logging
import numbers

import typing
import weakref
from enum import Enum

from PyQt5 import QtCore, QtWidgets, Qt
import matplotlib
# Make sure that we are using QT5
import stonesoup
from stonesoup.types.array import StateVector

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa E402
from matplotlib.figure import Figure  # noqa E402


LOG_ID = 'pairm_gui'

LOG = logging.getLogger()
ROW_HEIGHT_PIXELS = 17
SPINBOX_SIGNIFICANT_FIGURES = 4


class LogFunction:
    """Logging decorator that allows you to log with a
    specific logger.
    """
    # Customize these messages
    ENTRY_MESSAGE = 'Entering {}'
    EXIT_MESSAGE = 'Exiting {}'

    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, func):
        """Returns a wrapper that wraps func.
        The wrapper will log the entry and exit points of the function
        with logging.INFO level.
        """
        # set logger if it was not set earlier
        if not self.logger:
            logging.basicConfig()
            self.logger = LOG

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.logger.debug(self.ENTRY_MESSAGE.format(func.__name__))
            f_result = func(*args, **kwds)
            self.logger.debug(self.EXIT_MESSAGE.format(func.__name__))
            return f_result
        return wrapper


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        self.axes.text(0.5, 0.5, 'Click "Calculate" to show results', horizontalalignment='center',
                       verticalalignment='center')


class OneDNumericModel(QtCore.QAbstractListModel):
    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, **kwargs)
        if data is None:
            data = StateVector([])
        self._data = data

    def flags(self, index: QtCore.QModelIndex) -> Qt.Qt.ItemFlags:
        return Qt.Qt.ItemFlags(Qt.Qt.ItemIsSelectable | Qt.Qt.ItemIsEditable | Qt.Qt.ItemIsEnabled)

    def data(self, index: QtCore.QModelIndex, role: int = Qt.Qt.DisplayRole) -> typing.Any:
        # print(f'Getting _data index: {index}, role: {role} (Display is {Qt.Qt.DisplayRole})')
        if role in (Qt.Qt.DisplayRole, Qt.Qt.EditRole):
            # See below for the _data structure.
            return str(self._data[index.row()])

    def setData(self, index: QtCore.QModelIndex, value: typing.Any,
                role: int = Qt.Qt.DisplayRole) -> bool:
        assert index.column() == 0
        self._data[index.row()] = value
        self.dataChanged.emit(index, index)
        return True

    def rowCount(self, parent: QtCore.QModelIndex = None, *args, **kwargs) -> int:
        length = len(self._data)
        # print(length)
        return length


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


def prettify(string: str):
    return string.replace('_', ' ').title()


# noinspection PyCompatibility
class PropertyWidgetBuilder:
    """
    Class variable preferred_units is a dict containing property names and preferred display
    units, and a conversion factor for example if my_angle is stored in radians, but would more
    naturally displayed in degrees and bandwidth is stored in Hertz, but you would like to display
    in MHz. Can also be set with a conversion factor of one to just set the display unit.
    preferred_units = {'my_angle': ('deg', 180/pi),
                       'bandwidth': ('MHz', 1e6),
                       'power': ('W', 1)}
    Note this currently only uses property name for indexing. It could be extended to use class
    name as well, by making the key a tuple.
    In my case this is read in from a config file.
    """
    preferred_units = {}

    @LogFunction()
    def __init__(self, parent: QtWidgets.QWidget, model: stonesoup.types.base.Base,
                 property_name: str, property_: stonesoup.base.Property):

        self.model = model
        self.property_name = property_name
        self.doc = property_.doc
        self.property_model = None
        unit, self.conversion_factor = self.preferred_units.get(self.property_name, (None, 1))
        self.display = True

        LOG.debug(property_name)

        self.cls = property_.cls
        if self._is_not_set:
            self.widget = QtWidgets.QLabel('-')
        elif self.isnumeric:
            if self.cls is int:
                self.widget = QtWidgets.QSpinBox(parent=parent)
            else:

                self.widget = QtWidgets.QDoubleSpinBox(parent=parent)
                self.widget.setDecimals(self._calculate_decimal_places())
                self.widget.setSingleStep(self._calculate_step_size())
                self.widget.setMaximum(self.value*10)
            if unit:
                LOG.debug(f'{property_name}: setting units to {unit}')
                self.widget.setSuffix(' ' + unit)
            LOG.debug(f'{property_name}: connecting valueChanged on numeric')
            if not self._is_readonly:
                # noinspection PyUnresolvedReferences
                self.widget.valueChanged.connect(self.on_value_change)
            self.widget.setValue(self.value)
            LOG.debug(f'{property_name}: finished getting text from model')
        elif self.cls in (typing.List[int], typing.Sequence[int], StateVector, typing.List[float]):
            self.widget = QtWidgets.QListView(parent=parent)

            self.widget.setMaximumHeight(ROW_HEIGHT_PIXELS * len(self.value)
                                         + 2 * self.widget.frameWidth())
            if self.cls in (typing.List[int], typing.Sequence[int]):
                self.widget.setItemDelegate(SpinBoxDelegate())
            self.property_model = OneDNumericModel(data=self.value)
            self.widget.setModel(self.property_model)
        elif self.cls is bool:
            self.widget = QtWidgets.QCheckBox()
            self.widget.stateChanged.connect(self.on_state_changed)
            self.widget.setChecked(self.value)
        elif self._is_enum:
            self.widget = QtWidgets.QComboBox()
            for entry in self.cls:
                self.widget.addItem(entry.name, entry)
            # noinspection PyUnresolvedReferences
            self.widget.currentIndexChanged.connect(self.on_index_change)
            self.widget.setCurrentIndex(self.widget.findData(self.value))
        elif self.cls is weakref.ref:
            # weakrefs are not useful to display, but also don't want to raise warnings.
            self.widget = None
            self.display = False
        else:
            self.widget = None
            LOG.warning(f'Class not implemented: {self.cls} for property {self.property_name}')

        if self.widget is not None:
            if self._is_readonly:
                self.widget.setEnabled(False)
            self.widget.setToolTip(self.doc)

    def _calculate_decimal_places(self) -> int:
        dec = decimal.Decimal(self.value)
        decimal_point_position = dec.adjusted()
        return SPINBOX_SIGNIFICANT_FIGURES - decimal_point_position - 1

    def _calculate_step_size(self) -> float:
        return 10 ** (self._calculate_decimal_places() - 1)

    @property
    def _is_readonly(self):
        readonly = False
        try:
            test_value = self.value
            self.set(test_value)
        except AttributeError as err:
            if err.args[0].endswith('is readonly'):
                readonly = True
            else:
                raise
        return readonly

    @property
    def isnumeric(self):
        try:
            return issubclass(self.cls, numbers.Number)
        except TypeError:
            return False

    @property
    def value(self):
        value = getattr(self.model, self.property_name)
        if self.isnumeric and value is not None:
            value /= self.conversion_factor
        return value

    @property
    def _is_not_set(self):
        return self.value is None

    @property
    def _is_enum(self):
        try:
            return issubclass(self.cls, Enum)
        except TypeError:
            return False

    # @LogFunction()
    def make_label(self):
        if self.display:
            label = QtWidgets.QLabel(prettify(self.property_name))
            label.setToolTip(self.doc)
        else:
            label = None
        return label

    # @LogFunction()
    def on_value_change(self):
        self.set(self.widget.value())

    def on_state_changed(self):
        self.set(self.widget.isChecked())

    def on_index_change(self, index: int):
        self.set(self.widget.itemData(index))

    def set(self, value: typing.Any):
        if self.isnumeric and value is not None:
            value *= self.conversion_factor
        setattr(self.model, self.property_name, value)


class StoneSoupWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None, obj: stonesoup.types.base.Base = None, header: str = ''):
        if header == '':
            header = obj.__class__.__name__
        QtWidgets.QGroupBox.__init__(self, header, parent=parent)
        self.widgets = []
        self.children = []
        # noinspection PyProtectedMember
        for prop_name, prop in obj._properties.items():
            try:
                is_stone_soup_object = (issubclass(prop.cls, stonesoup.types.base.Base)
                                        and getattr(obj, prop_name) is not None)
            except TypeError:
                is_stone_soup_object = False
            if is_stone_soup_object:
                self.children.append(StoneSoupWidget(parent=self, obj=getattr(obj, prop_name),
                                                     header=prettify(prop_name)))
            else:
                widget = PropertyWidgetBuilder(self, obj, prop_name, prop)
                self.widgets.append(widget)

        direct_layout = QtWidgets.QFormLayout()

        if self.children:
            self.setLayout(QtWidgets.QHBoxLayout())
            self.layout().addLayout(direct_layout)
            child_layout = QtWidgets.QVBoxLayout()
            self.layout().addLayout(child_layout)
            for child in self.children:
                child_layout.addWidget(child)
        else:
            self.setLayout(direct_layout)

        for widget in self.widgets:
            direct_layout.addRow(widget.make_label(), widget.widget)
