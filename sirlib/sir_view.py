""" code-behind for sir thread view port """
# pylint: disable=E0611
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from cv2 import (
    circle, cvtColor, polylines,
    COLOR_GRAY2BGR, COLOR_GRAY2RGB, COLOR_BGR2RGB)

CREATOR_FILE = 'sirlib/sir_view.ui'
SIR_VIEW, QT_BASE_CLASS = uic.loadUiType(CREATOR_FILE)


class SIRView(QWidget, SIR_VIEW):
    """ The a GUI widget for monitoring SIR threads """

    def __init__(self, parent=None):
        super(SIRView, self).__init__(parent)
        SIR_VIEW.__init__(self)
        self.setupUi(self)
        self.tracker = None

        self.btn_pause.clicked.connect(self.do_pause)
        self.btn_abort.clicked.connect(self.do_abort)

    @pyqtSlot(dict)
    def change_frame(self, frame_details):
        """ slot called when frame changes """

        estimate = frame_details['estimate']

        # plot over frame and display
        colorized = cvtColor(frame_details['frame'], COLOR_GRAY2BGR)
        if not np.any(np.isnan(estimate)):
            # template half-width half-height
            hh = self.tracker.template_height/2.0
            hw = self.tracker.template_width/2.0
            blue = (255, 0, 0)
            yellow = (0, 255, 255)

            gt = np.array(frame_details['gt'], dtype=np.float32)
            self.plot_X_box([gt[0], gt[1], 0, 0, 1.0, 0],
                            gt[3], gt[2], colorized, blue)
            circle(colorized, (gt[1], gt[0]), 5, blue)
            self.plot_X_box(estimate, hw, hh, colorized, yellow)
            circle(colorized, (estimate[1], estimate[0]), 5, yellow)
        colorized = cvtColor(colorized, COLOR_BGR2RGB)
        qt_stride = int(colorized.nbytes/colorized.shape[0])
        qt_image = QImage(colorized.data, colorized.shape[1],
                          colorized.shape[0], qt_stride,
                          QImage.Format_RGB888)
        self.lbl_frame.setPixmap(QPixmap(qt_image))

        frame_details = (
            f"Frame number: {frame_details['frame_number']}\n"
            f"Estimate: ("
            f"{estimate[0]:6.1f}, "
            f"{estimate[1]:6.1f}, "
            f"{estimate[2]:6.1f}, "
            f"{estimate[3]:6.1f}, "
            f"{estimate[4]:6.1f}, "
            f"{estimate[5]:6.1f})\n"
            f"Error: {np.linalg.norm(frame_details['error'])}\n"
            f"Neff: {frame_details['neff']}"
        )
        self.lbl_frame_details.setText(frame_details)

    def plot_X_box(self, X, hw, hh, colorized, color):
        # transform and translate estimate
        tm = X[0]
        tn = X[1]
        m = X[4]
        c = np.cos(X[5])
        s = np.sin(X[5])
        est_pts = np.array(
            [
                [
                    # upper-left (-hh,-hw)
                    m*(-hw*c - hh*s) + tn,
                    m*(hw*s - hh*c) + tm
                ],
                [
                    # upper-right (-hh,hw)
                    m*(-hw*c + hh*s) + tn,
                    m*(hw*s + hh*c) + tm
                ],
                [
                    # lower-right (hh,hw)
                    m*(hw*c + hh*s) + tn,
                    m*(-hw*s + hh*c) + tm
                ],
                [
                    # lower-left
                    m*(hw*c - hh*s) + tn,
                    m*(-hw*s - hh*c) + tm
                ]
            ], np.int32)

        est_pts = est_pts.reshape((-1, 1, 2))
        polylines(colorized, [est_pts], isClosed=True, thickness=2, color=color)

    @pyqtSlot(object)
    def change_template(self, extracted_template):
        """ slot called when template changes """
        extracted_template *= 255.0/extracted_template.max()
        extracted_template = extracted_template.astype(np.uint8)

        #  grayscale bgr is rgb
        colorized = cvtColor(extracted_template, COLOR_GRAY2RGB)
        qt_stride = int(colorized.nbytes/colorized.shape[0])
        qt_image = QImage(
            colorized.data, colorized.shape[1],
            colorized.shape[0], qt_stride, QImage.Format_RGB888)
        self.lbl_template.setPixmap(QPixmap(qt_image))

    @pyqtSlot()
    def tracker_finished(self):
        """ slot called when tracker thread finishes """
        pass

    @pyqtSlot()
    def do_pause(self):
        """ perform actions when pausing """
        if self.tracker.paused:
            self.btn_pause.setText('Resume')
        else:
            self.btn_pause.setText('Pause')

    @pyqtSlot()
    def do_abort(self):
        """ abort run """
        pass

    def attach_tracker(self, sir_thread):
        """ attach sir thread """
        self.detach_tracker()
        self.tracker = sir_thread
        self.btn_pause.clicked.connect(self.tracker.do_pause)
        self.tracker.signals.template_changed.connect(self.change_template)
        self.tracker.signals.frame_changed.connect(self.change_frame)
        self.tracker.signals.finished.connect(self.tracker_finished)

        # update tracker details
        tracker_details = (
            f"Sequence name: {self.tracker.job_options['name']}\n"
            f"Job ID: {self.tracker.job_options['job_id']}\n"
            f"Start Frame: {self.tracker.job_options['start_frame']}\n"
            f"End Frame: {self.tracker.job_options['end_frame']}\n"
            f"Particle count: {self.tracker.job_options['particle_count']}\n"
            f"Score type: {self.tracker.job_options['score_type']}\n"
            f"Filter mode: {self.tracker.job_options['filter_mode']}\n"
            f"Update interval: {self.tracker.job_options['update_interval']}\n"
            f"Update method: {self.tracker.job_options['update_method']}\n"
            f"History lenght: "
            f"{self.tracker.job_options['historical_length']}\n"
        )
        self.lbl_tracker_details.setText(tracker_details)

    def detach_tracker(self):
        """ detach sir thread """
        if self.tracker is None:
            return
        self.btn_pause.clicked.disconnect(self.tracker.do_pause)
        self.tracker.signals.frame_changed.disconnect(self.change_frame)
        self.tracker.signals.template_changed.disconnect(self.change_template)
        self.tracker.signals.finished.disconnect(self.tracker_finished)
        self.tracker = None

# for testing widget layout
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication  # pylint: disable=C0412
    APP = QApplication(sys.argv)
    VIEWER_APP = SIRView()
    VIEWER_APP.show()
    sys.exit(APP.exec_())
