import decimal
import numbers

import weakref
from datetime import datetime
from enum import Enum
from typing import List, Sequence, Any

import numpy as np

from stonesoup.base import Base
from stonesoup.gui.gui_logging import LOG, LogFunction
from stonesoup.gui.models import StoneSoupModel, OneDNumericModel, NDArrayModel, SpinBoxDelegate
from .qt import QtWidgets, QtCore
import stonesoup
from stonesoup.types.array import StateVector

import matplotlib
# Make sure that we are using QT5
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # noqa E402
from matplotlib.figure import Figure  # noqa E402

ROW_HEIGHT_PIXELS = 17
SPINBOX_SIGNIFICANT_FIGURES = 4


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
    def __init__(self, parent: QtWidgets.QWidget, model: StoneSoupModel,
                 property_name: str, property_: stonesoup.base.Property):

        self.model = model
        self.property_name = property_name
        self.doc = property_.doc
        self.property_model = None
        self.unit, self.conversion_factor = self.preferred_units.get(self.property_name, (None, 1))
        self.display = True
        self.parent = parent
        LOG.debug(property_name)

        self.cls = property_.cls
        self.widget = None
        self._build_widget()

        if self.widget is not None:
            if self._is_readonly:
                self.widget.setEnabled(False)
            self.widget.setToolTip(self.doc)
        self.update_value()

    def _build_widget(self):
        old_widget = self.widget
        if self._is_not_set:
            self.widget = QtWidgets.QLabel('-')
        elif self.isnumeric:
            if self.cls is int:
                self.widget = QtWidgets.QSpinBox(parent=self.parent)
            else:
                self.widget = QtWidgets.QDoubleSpinBox(parent=self.parent)
                self.widget.setDecimals(self._calculate_decimal_places())
                self.widget.setSingleStep(self._calculate_step_size())
            abs_max = np.abs(self.value) * 1000
            if abs_max == 0:
                abs_max = 1000
            self.widget.setMaximum(abs_max)
            self.widget.setMinimum(-abs_max)
            if self.unit:
                LOG.debug(f'{self.property_name}: setting units to {self.unit}')
                # noinspection PyTypeChecker
                self.widget.setSuffix(' ' + self.unit)
            LOG.debug(f'{self.property_name}: connecting valueChanged on numeric')
            if not self._is_readonly:
                # noinspection PyUnresolvedReferences
                self.widget.valueChanged.connect(self.on_value_change)
            self.widget.setValue(self.value)
            LOG.debug(f'{self.property_name}: finished getting text from model')
        elif self._is_numeric_sequence(self.cls):
            self.widget = QtWidgets.QListView(parent=self.parent)

            self.widget.setMaximumHeight(ROW_HEIGHT_PIXELS * len(self.value)
                                         + 2 * self.widget.frameWidth())
            if self.cls in (List[int], Sequence[int]):
                self.widget.setItemDelegate(SpinBoxDelegate())
            self.property_model = OneDNumericModel(data=self.value)
            self.widget.setModel(self.property_model)
        elif self.cls is bool:
            self.widget = QtWidgets.QCheckBox()
            self.widget.stateChanged.connect(self.on_state_changed)
        elif self._is_enum:
            self.widget = QtWidgets.QComboBox()
            for entry in self.cls:
                self.widget.addItem(entry.name, entry)
            # noinspection PyUnresolvedReferences
            self.widget.currentIndexChanged.connect(self.on_index_change)
        elif self.cls is np.ndarray:
            self.widget = QtWidgets.QTableView()
            self.property_model = NDArrayModel(array=self.value)
            self.widget.setModel(self.property_model)
        elif self.cls is datetime:
            self.widget = QtWidgets.QLabel()
        elif self.cls is weakref.ref:
            # weakrefs are not useful to display, but also don't want to raise warnings.
            self.widget = None
            self.display = False
        else:
            self.widget = None
            LOG.warning(f'Class not implemented: {self.cls} for property {self.property_name}')
        if old_widget is not None:
            containing_layout = old_widget.parent().layout()
            containing_layout.replaceWidget(old_widget, self.widget)
            old_widget.deleteLater()

    @staticmethod
    def _is_numeric_sequence(cls: type) -> bool:
        if cls in (List[int], Sequence[int], StateVector, List[float], Sequence[float]):
            return True
        # the hack below needed because [int] is a list of length 1 containing the types int,
        # not the type "list of int"
        try:
            if isinstance(cls, list) and issubclass(cls[0], numbers.Number):
                return True
        except TypeError:
            return False
        return False

    def _calculate_decimal_places(self) -> int:
        dec = decimal.Decimal(self.value)
        if dec == 0:
            return SPINBOX_SIGNIFICANT_FIGURES
        decimal_point_position = dec.adjusted()
        return SPINBOX_SIGNIFICANT_FIGURES - decimal_point_position - 1

    def _calculate_step_size(self) -> float:
        return 10 ** -(self._calculate_decimal_places() - 1)

    @property
    def _is_readonly(self):
        readonly = False
        try:
            test_value = self.value
            self.set(test_value)
        except AttributeError as err:
            msg = err.args[0]
            if (msg.endswith('is readonly')
                    or msg == "can't set attribute"
                    or msg.startswith('Cannot set')):
                readonly = True
            else:
                raise
        return readonly

    @property
    def isnumeric(self):
        try:
            return (issubclass(self.cls, numbers.Number)
                    and not self._is_enum
                    and not issubclass(self.cls, bool))
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

    def set(self, value: Any):
        if self.isnumeric and value is not None:
            value *= self.conversion_factor
        setattr(self.model, self.property_name, value)

    def update_value(self):
        if isinstance(self.widget, QtWidgets.QLabel) and not self._is_not_set:
            self._build_widget()

        if self._is_not_set:
            self.widget = QtWidgets.QLabel('-')
        elif self.isnumeric:
            self.widget.setValue(self.value)
        elif self._is_numeric_sequence(self.cls) or self.cls is np.ndarray:
            self.property_model.resetData(self.value)
        elif self.cls is bool:
            self.widget.setChecked(self.value)
        elif self.cls is datetime:
            self.widget.setText(str(self.value))
        elif self._is_enum:
            self.widget.setCurrentIndex(self.widget.findData(self.value))


class StoneSoupWidget(QtWidgets.QGroupBox):
    # noinspection PyUnresolvedReferences
    def __init__(self, parent=None, obj: Base = None, header: str = ''):
        if header == '':
            header = obj.__class__.__name__
        QtWidgets.QGroupBox.__init__(self, header, parent=parent)
        self.model = StoneSoupModel(self, obj)
        self.model.dataChanged.connect(self.update_values)
        self.widgets = []
        self.children = []
        try:
            hidden_properties = self.model.obj.__class__.hidden_properties
        except AttributeError:
            hidden_properties = []
        # noinspection PyProtectedMember
        properties = self.model._properties
        try:
            extra_properties = self.model.obj.__class__.extra_properties
        except AttributeError:
            extra_properties = {}
        properties.update(extra_properties)

        for prop_name, prop in properties.items():
            if prop_name in hidden_properties:
                continue
            try:
                is_stone_soup_object = self._is_stone_soup_object(prop.cls, prop_name)
            except TypeError:
                is_stone_soup_object = False
            try:
                is_sequence_of_stone_soup_objects = (isinstance(prop.cls, list)
                                                     and self._is_stone_soup_object(prop.cls[0],
                                                                                    prop_name))
            except TypeError:
                is_sequence_of_stone_soup_objects = False
            if is_stone_soup_object:
                self.children.append(StoneSoupWidget(parent=self, obj=getattr(self.model,
                                                                              prop_name),
                                                     header=prettify(prop_name)))
            elif is_sequence_of_stone_soup_objects:
                list_widget = QtWidgets.QGroupBox(prettify(prop_name))
                list_widget.setLayout(QtWidgets.QVBoxLayout())
                for list_entry in getattr(self.model, prop_name):
                    list_widget.layout().addWidget(StoneSoupWidget(parent=self, obj=list_entry))
                self.children.append(list_widget)
            else:
                widget = PropertyWidgetBuilder(self, self.model, prop_name, prop)
                self.widgets.append(widget)

        direct_layout = QtWidgets.QFormLayout()
        direct_widget = QtWidgets.QWidget()
        direct_widget.setLayout(direct_layout)
        self.setLayout(QtWidgets.QHBoxLayout())
        if self.widgets:
            self.layout().addWidget(direct_widget)

        if self.children:
            child_widget = QtWidgets.QWidget()

            child_layout = QtWidgets.QVBoxLayout()
            child_widget.setLayout(child_layout)
            for child in self.children:
                child_layout.addWidget(child)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidget(child_widget)
            scroll.setWidgetResizable(True)
            scroll.setMinimumWidth(200)
            scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            # scroll.setFrameStyle(QtWidgets.QFrame.NoFrame)
            self.layout().addWidget(scroll)

        for widget in self.widgets:
            direct_layout.addRow(widget.make_label(), widget.widget)

        # scroll.setMinimumWidth(direct_widget.minimumWidth()
        #                        + 2 * scroll.frameWidth()
        #                        + scroll.verticalScrollBar().sizeHint().width())

    def _is_stone_soup_object(self, cls, prop_name):
        val = getattr(self.model, prop_name)
        return issubclass(cls, Base) and bool(val)

    def update_values(self):
        for widget in self.widgets:
            widget.update_value()
