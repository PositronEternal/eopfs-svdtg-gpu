""" sir particle filter target tracker """
from os import path, makedirs
import json
import gc
import numpy as np
import tensorflow as tf
# pylint: disable=E0611
from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout
import time
from sir_graph import SIRGraph
from mcvideo import MCVideo
from sir_view import SIRView


class SIRTrackerSignals(QObject):
    status_changed = pyqtSignal(int, str)
    frame_changed = pyqtSignal(dict)
    template_changed = pyqtSignal(object)
    finished = pyqtSignal(int)


class SIRTracker(QRunnable):
    def __init__(
            self,
            job_options):
        super(SIRTracker, self).__init__()
        self.job_options = job_options
        self.paused = False
        self._graph = None
        self._video = None
        self._result_file = None
        self.signals = SIRTrackerSignals()

    def init_results(self):
        """ initialize memory and file save path for results """
        result_path = path.join(
            self.job_options['save_path'],
            self.job_options['name'],
            (str(self.job_options['start_frame']) +
                '_' + str(self.job_options['end_frame'])),
            'pc_' + str(self.job_options['particle_count']),
            self.job_options['score_type'],
            self.job_options['filter_mode'],
            'ui_' + str(self.job_options['update_interval']),
            self.job_options['update_method'],
            'hl_' + str(self.job_options['historical_length'])
        )

        if not path.exists(result_path):
            makedirs(result_path)

        self._result_file = path.join(
            result_path,
            'results_' + str(self.job_options['run']) + '.json',
        )

        self.results = {
            'job_options': self.job_options,
            'frame_number': [],
            'estimate': [],
            'error': [],
            'neff': [],
            'template_updated': []
        }

    def do_pause(self):
        """ pause/unpause thread """
        if not self.paused:
            self.paused = True
            self.signals.status_changed.emit(
                self.job_options['job_id'], 'Paused')
        else:
            self.paused = False
            self.signals.status_changed.emit(
                self.job_options['job_id'], 'Running')

    def run(self):
        """ main execution method for tracker """

        self.signals.status_changed.emit(
            self.job_options['job_id'], 'Loading sequence')
        self.load_sequence()

        if self.job_options['save_path'] is not None:
            self.init_results()
            self.signals.frame_changed.connect(self.on_frame_change)
            self.signals.finished.connect(self.on_finished)

        # skip if result for job exists
        if self._result_file is not None and \
                path.exists(self._result_file):
            self._video = None
            # prevent overwriting existing results
            self.signals.finished.disconnect(self.on_finished)
            self.signals.status_changed.emit(
                self.job_options['job_id'],
                'Skipped')
            self.signals.finished.emit(self.job_options['job_id'])
            return

        self.signals.status_changed.emit(
            self.job_options['job_id'], 'Generating Graph')
        self.load_graph()

        self.signals.status_changed.emit(
            self.job_options['job_id'], 'Running')

        with tf.Session(graph=self._graph.graph) as sess:
            sess.run(tf.global_variables_initializer())

            if self.job_options['filter_mode'] == 'AUX':
                filter_fn = self.fn_filter_aux
            elif self.job_options['filter_mode'] == 'RESAMPLE':
                filter_fn = self.fn_filter_resample
            else:
                filter_fn = self.fn_filter_none

            # main tracker loop
            for frame_num in range(
                    self.job_options['start_frame'],
                    self.job_options['end_frame']):
                while self.paused:
                    pass

                gtc = self._video.get_gt_center(frame_num)

                # seed particles
                if frame_num == self.job_options['start_frame']:
                    seed_x = np.tile(
                        np.array([gtc[0], gtc[1], 0., 0., 1., 0.]),
                        [self.job_options['particle_count'], 1])
                    sess.run(
                        self._graph.seed_p,
                        feed_dict={self._graph.p_seed: seed_x})
                    self._graph.set_template_roi(sess, gtc[0], gtc[1], 1, 0)
                    sess.run(self._graph.reset_w)

                frame_details = {}
                frame_details['frame_number'] = frame_num

                pix_frame = self._video.get_pix_frame(frame_num)
                sess.run(
                    self._graph.set_frame,
                    feed_dict={self._graph.frame_input: pix_frame})
                frame_details['frame'] = pix_frame

                do_update = \
                    self.job_options['update_interval'] > 0 and \
                    frame_num % self.job_options['update_interval'] == 0
                frame_details['template_updated'] = do_update

                self._graph.maintain_template(sess, do_update)
                if do_update or frame_num == 0:
                    extracted_template = sess.run(self._graph.template)
                    self.signals.template_changed.emit(extracted_template)

                filter_fn(sess)
                np_estimate = sess.run(self._graph.estimate)
                frame_details['estimate'] = np_estimate
                frame_details['error'] = \
                    np_estimate[0:2] - [gtc[0], gtc[1]]
                frame_details['neff'] = sess.run(self._graph.neff)

                self.signals.frame_changed.emit(frame_details)

        self._graph = None
        self._video = None
        # force garbage collect to remove unused tensorflow graphs
        gc.collect()
        self.signals.status_changed.emit(
            self.job_options['job_id'], 'Complete')
        self.signals.finished.emit(self.job_options['job_id'])

    def load_sequence(self):
        """ load video sequence and ground truth source """

        seq_name = self.job_options['name']
        root_path = self.job_options['root_path']

        seq_path = path.join(root_path, seq_name)
        pix_path = path.join(seq_path, 'frames_' + seq_name + '.bin')
        mod_path = path.join(seq_path, 'amfm_' + seq_name + '.bin')
        gt_path = path.join(seq_path, 'video_params_' + seq_name + '.mat')
        self._video = MCVideo(pix_path, mod_path, gt_path)

        # generate actual end frame and replace if necessary
        end_frame = self.job_options['end_frame']
        if end_frame < 0 or end_frame > self._video.length:
            self.job_options['end_frame'] = self._video.length-1

    def load_graph(self):
        sir_options = {s: self.job_options[s] for s in
                       ('particle_count', 'score_type', 'filter_mode',
                        'update_interval', 'update_method',
                        'historical_length', 'seed')}

        video_options = {}
        video_options['height'] = self._video.height
        video_options['width'] = self._video.width

        template_options = {}
        # pylint: disable=E0633
        template_options['width'], template_options['height'] = \
            np.int32(
                self._video.get_gt_tsize(
                    self.job_options['start_frame']))

        self._graph = SIRGraph(sir_options, video_options, template_options)
        return sir_options, video_options, template_options

    def on_frame_change(self, frame_details):
        """ save results to memory """
        self.results['frame_number'].append(frame_details['frame_number'])
        self.results['estimate'].append(frame_details['estimate'].tolist())
        self.results['error'].append(frame_details['error'].tolist())
        self.results['neff'].append(frame_details['neff'].tolist())
        self.results['template_updated'].append(
            frame_details['template_updated'])

    def on_finished(self):
        """ save results """
        with open(self._result_file, 'w') as f:
            json.dump(self.results, f)

    #  The filters, these are the SIR tracking algorithm routinesfunctions
    #  that utilize the tensorflow graph
    # to perform the

    def fn_filter_none(self, session):
        self._graph.shift_energies(session)
        session.run(self._graph.predict_from_p)
        self._graph.store_energies(session)
        session.run(self._graph.store_score)
        session.run(self._graph.update_w)

    def fn_filter_resample(self, session):
        session.run(self._graph.store_ridx)
        session.run(self._graph.resample_p)
        if self.job_options['score_type'] == 'ASV' or \
                self.job_options['score_type'] == 'ASVHO':
            session.run(self._graph.resample_e1_template_ss)
            session.run(self._graph.resample_e1_ss)
        if self.job_options['score_type'] == 'ASVHO':
            session.run(self._graph.resample_e2_template_ss)
            session.run(self._graph.resample_e2_ss)
        session.run(self._graph.reset_w)
        session.run(self._graph.predict_from_p)
        self._graph.store_energies(session)
        session.run(self._graph.store_score)
        session.run(self._graph.update_w)

    def fn_filter_aux(self, session):
        self._graph.shift_energies(session)
        session.run(self._graph.store_aux_p)
        session.run(self._graph.predict_from_p)
        self._graph.store_energies(session)
        session.run(self._graph.store_score)
        session.run(self._graph.update_w)
        session.run(self._graph.store_ridx)
        session.run(self._graph.restore_p_from_aux)
        session.run(self._graph.resample_p)
        if self.job_options['score_type'] == 'ASV' or \
                self.job_options['score_type'] == 'ASVHO':
            session.run(self._graph.resample_e1_template_ss)
            session.run(self._graph.resample_e1_ss)
        if self.job_options['score_type'] == 'ASVHO':
            session.run(self._graph.resample_e2_template_ss)
            session.run(self._graph.resample_e2_ss)
        session.run(self._graph.reset_w)
        session.run(self._graph.predict_from_p)
        self._graph.store_energies(session)
        session.run(self._graph.store_score)
        session.run(self._graph.update_w)


class SIRWindow(QMainWindow):
    def __init__(self, parent=None):
        super(SIRWindow, self).__init__(parent)
        self._closing = False

        self.setWindowTitle('SIR Tracker View')
        self.setGeometry(300, 600, 300, 300)
        self.view = SIRView(self)
        self.setCentralWidget(self.view)

    def closeEvent(self, *args, **kwargs):
        self.closing = True

    def attach_tracker(self, tracker):
        self.tracker = tracker
        self.view.attach_tracker(tracker)

    def detach_tracker(self):
        self.view.detach_tracker()
        self.tracker = None

    def on_finished(self, job_id):
        self.view.detach_tracker()


if __name__ == '__main__':
    import sys
    from PyQt5.QtCore import QThreadPool
    from PyQt5.QtWidgets import QApplication

    APP = QApplication(sys.argv)
    SIRWINDOW = SIRWindow()
    THREADPOOL = QThreadPool()
    job_options1 = {
        'job_id': 1,
        'seed': 666,
        'root_path': '/mnt/data/processedsequences',
        'name': 'Car4',
        'start_frame': 0,
        'end_frame': 25,
        "save_path": None,  # '/mnt/data/results',
        "particle_count": 300,
        "score_type": 'ASVHO',
        "filter_mode": 'AUX',
        "update_interval": 20,
        "update_method": 'SCORE',
        "historical_length": 10,
        "run": 0
    }
    worker1 = SIRTracker(job_options1)
    SIRWINDOW.attach_tracker(worker1)
    SIRWINDOW.show()
    THREADPOOL.start(worker1)
    sys.exit(APP.exec_())
