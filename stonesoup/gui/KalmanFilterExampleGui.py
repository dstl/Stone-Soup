import logging
import os
import sys
from datetime import datetime, timedelta

from PyQt5 import QtWidgets, QtCore
import numpy as np

from stonesoup.gui.stonesoup_widgets import MplCanvas, StoneSoupWidget
from stonesoup.gui import stonesoup_widgets
from stonesoup.models.transition.linear import ConstantVelocity, \
    CombinedLinearGaussianTransitionModel
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Stonesoup GUI demo")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.file_quit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)
        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)
        self.help_menu.addAction('&About', self.about)

        self.main_widget = QtWidgets.QWidget(self)

        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                                       ConstantVelocity(0.05)])
        self.predictor = KalmanPredictor(self.transition_model)

        self.main_widget.setLayout(QtWidgets.QHBoxLayout())
        self.main_widget.layout().addWidget(StoneSoupWidget(self.main_widget, self.predictor), 1)
        scenario_widget = QtWidgets.QTabWidget(self.main_widget)
        scenario_widget.addTab(DisplayWidget(), 'Kalman Filter Demo')
        # scenario_widget.addTab(SanitiseWidget(radar=self.radar), 'Sanitise Airspace')
        self.main_widget.layout().addWidget(scenario_widget, 10)
        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

    def file_quit(self):
        self.close()

    # noinspection PyUnusedLocal
    def close_event(self, ce):
        self.file_quit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """Air intercept model GUI
Crown Copyright 2020
Written by Edward Rogers

"""
                                    )


class ScenarioWidget(QtWidgets.QGroupBox):
    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__('Scenario', parent)

        self.setLayout(QtWidgets.QFormLayout())
        self.time_steps_widget = QtWidgets.QSpinBox(self)
        self.time_steps_widget.setMaximum(100)
        self.time_steps_widget.setValue(21)
        self.layout().addRow('Number of time steps', self.time_steps_widget)

        # self.closing_speed_widget = QtWidgets.QDoubleSpinBox(self)
        # self.closing_speed_widget.setMaximum(10000)
        # self.closing_speed_widget.setValue(1185)
        # self.closing_speed_widget.setSuffix('kts')
        # self.layout().addRow('Closing speed', self.closing_speed_widget)

        self.calculate_button = QtWidgets.QPushButton(text='Calculate', parent=self)
        self.layout().addRow('', self.calculate_button)


class DisplayWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget = None, radar=None):
        super().__init__(parent)
        self.radar = radar
        self.setLayout(QtWidgets.QHBoxLayout())
        self.canvas = MplCanvas()
        self.scenario_widget = ScenarioWidget(self)
        self.layout().addWidget(self.scenario_widget)
        # noinspection PyUnresolvedReferences
        self.scenario_widget.calculate_button.clicked.connect(self.do_simulation)
        self.layout().addWidget(self.canvas)

    def do_simulation(self):
        fig = self.canvas.figure
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.axis('equal')

        start_time = datetime.now()
        transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                                  ConstantVelocity(0.05)])
        truth = GroundTruthPath([GroundTruthState([0, 1, 0, 1], timestamp=start_time)])
        for k in range(1, self.scenario_widget.time_steps_widget.value()):
            truth.append(GroundTruthState(
                transition_model.function(truth[k - 1], noise=True,
                                          time_interval=timedelta(seconds=1)),
                timestamp=start_time + timedelta(seconds=k)))

        # Plot the result
        ax.plot([state.state_vector[0] for state in truth],
                [state.state_vector[2] for state in truth],
                linestyle="--")
        self.canvas.draw()


def main():
    logging.basicConfig()
    log = logging.getLogger(stonesoup_widgets.LOG_ID)
    log.setLevel(logging.INFO)
    log.info('Starting the GUI')

    q_app = QtWidgets.QApplication(sys.argv)
    aw = ApplicationWindow()
    aw.show()
    sys.exit(q_app.exec_())


if __name__ == '__main__':
    main()
