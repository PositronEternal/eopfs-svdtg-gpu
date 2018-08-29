""" run sir filters agains videos in batches for later analysis """
import sys
import json
# pylint: disable=E0611
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThreadPool, pyqtSignal, pyqtSlot, QModelIndex
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QFileDialog,
    QMessageBox)
from sir_tracker import SIRTracker, SIRWindow


BATCH_CREATOR_FILE = 'sirlib/sir_batch.ui'
BATCH_WIDGET, QT_BASE_CLASS = uic.loadUiType(BATCH_CREATOR_FILE)


class BatchWindow(QMainWindow, BATCH_WIDGET):
    """ The batch sir tracker application window """

    NAME, \
        RUN, \
        PARTICLE_COUNT, \
        SCORE_TYPE, \
        FILTER_MODE, \
        UPDATE_INTERVAL, \
        UPDATE_METHOD, \
        HISTORICAL_LENGTH, \
        STATUS = range(9)

    def __init__(self, parent=None):
        super(BatchWindow, self).__init__(parent)
        BATCH_WIDGET.__init__(self)
        self.setupUi(self)

        self.create_run_model(self)
        self.tv_runs.clicked.connect(self.on_tv_runs_clicked)
        self.btn_load.clicked.connect(self.do_load)
        self.btn_start.clicked.connect(self.do_start)
        self.btn_cancel.clicked.connect(self.do_cancel)
        self.options = {}
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.runs = None
        self.threadpool = QThreadPool()
        self.viewer_index = None
        self.viewer = SIRWindow(self)

    @pyqtSlot()
    def do_load(self):
        """ load batch parameters from a file """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "QfileDialog.getOpenFileName()",
            "",
            "Json Files (*.json)",
            options=options)
        if not filename:
            self.btn_start.setEnabled(False)
            return

        with open(filename, 'r') as bo_file:
            self.options = json.load(bo_file)

        batch_details = (
            f"Source root: {self.options['root_path']}\n"
            f"Save path: {self.options['save_path']}"
        )
        self.lbl_batch_details.setText(batch_details)

        self.runs = [
            {
                'job_options':
                {
                    'name': sequence['name'],
                    'start_frame': sequence['start_frame'],
                    'end_frame': sequence['end_frame'],
                    'particle_count': particle_count,
                    'score_type': score_type,
                    'filter_mode': filter_mode,
                    'update_interval': update_interval,
                    'update_method': update_method,
                    'historical_length': historical_length,
                    'run': run,
                    'root_path': self.options['root_path'],
                    'save_path': self.options['save_path'],
                },
                'status': 'Not started'
            }
            for sequence in self.options['sequences']
            for particle_count in self.options['particle_counts']
            for score_type in self.options['score_types']
            for filter_mode in self.options['filter_modes']
            for update_interval in self.options['update_intervals']
            for update_method in self.options['update_methods']
            for historical_length in self.options['historical_lengths']
            for run in range(self.options['number_runs'])
        ]

        # attach run ids, seed, and tracker objects
        for index, r in enumerate(self.runs):
            r['job_options']['job_id'] = index
            r['job_options']['seed'] = index
            r['tracker'] = SIRTracker(r['job_options'])
            r['tracker'].signals.status_changed.connect(self.on_status_change)

        # add to display
        self.clear_runs()
        for ro_idx in range(len(self.runs)):
            run_options = self.runs[ro_idx]['job_options']
            self.add_run(run_options, self.runs[ro_idx]['status'])
        self.btn_start.setEnabled(True)

    @pyqtSlot()
    def do_start(self):
        """ start button callback """
        self.btn_load.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        for r in self.runs:
            self.threadpool.start(r['tracker'])

    @pyqtSlot(int, str)
    def on_status_change(self, job_id, status):
        """ a job status has changed """
        self.update_status(job_id, status)

    @pyqtSlot()
    def do_cancel(self):
        """ cancel button callback """
        pass

    def create_run_model(self, parent):
        """ set model for the list(tree) of batch entries """
        self.model = QStandardItemModel(0, 9, parent)
        self.model.setHeaderData(self.NAME, Qt.Horizontal, "Name")
        self.model.setHeaderData(self.RUN, Qt.Horizontal, "Run Index")
        self.model.setHeaderData(
            self.PARTICLE_COUNT, Qt.Horizontal, "particles")
        self.model.setHeaderData(self.SCORE_TYPE, Qt.Horizontal, "Score Type")
        self.model.setHeaderData(
            self.FILTER_MODE, Qt.Horizontal, "Filter Mode")
        self.model.setHeaderData(self.UPDATE_INTERVAL,
                                 Qt.Horizontal, "Update Interval")
        self.model.setHeaderData(
            self.UPDATE_METHOD, Qt.Horizontal, "Update Method")
        self.model.setHeaderData(
            self.HISTORICAL_LENGTH, Qt.Horizontal, "History Length")
        self.model.setHeaderData(self.STATUS, Qt.Horizontal, "Status")
        self.tv_runs.setModel(self.model)

    def add_run(self, run_options, status):
        """ add a run to the list """
        row_idx = self.model.rowCount()
        self.model.insertRow(row_idx)
        self.model.setData(
            self.model.index(row_idx, self.NAME), run_options['name'])
        self.model.setData(
            self.model.index(row_idx, self.RUN), run_options['run'])
        self.model.setData(self.model.index(
            row_idx, self.PARTICLE_COUNT), run_options['particle_count'])
        self.model.setData(self.model.index(
            row_idx, self.SCORE_TYPE), run_options['score_type'])
        self.model.setData(self.model.index(
            row_idx, self.FILTER_MODE), run_options['filter_mode'])
        self.model.setData(self.model.index(
            row_idx, self.UPDATE_INTERVAL), run_options['update_interval'])
        self.model.setData(self.model.index(
            row_idx, self.UPDATE_METHOD), run_options['update_method'])
        self.model.setData(self.model.index(
            row_idx, self.HISTORICAL_LENGTH), run_options['historical_length'])
        self.model.setData(self.model.index(row_idx, self.STATUS), status)

    def update_status(self, index, status):
        """ update status of a run given its index """
        self.runs[index]['status'] = status
        self.model.setData(self.model.index(index, self.STATUS), status)

    def clear_runs(self):
        """ clear run list """
        self.model.removeRows(0, self.model.rowCount())

    @pyqtSlot(QModelIndex)
    def on_tv_runs_clicked(self, index):
        """ list selection changed """
        row_idx = index.row()

        if self.viewer_index is None:
            self.viewer.show()

        self.viewer_index = row_idx
        self.viewer.attach_tracker(self.runs[row_idx]['tracker'])


if __name__ == '__main__':
    BATCH_APP = QApplication(sys.argv)
    BATCH_VIEWER_APP = BatchWindow()
    BATCH_VIEWER_APP.show()
    sys.exit(BATCH_APP.exec_())
