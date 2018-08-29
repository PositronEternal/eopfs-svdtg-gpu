""" sir filter graph container with functions to interact with it """
import numpy as np
import tensorflow as tf
# pylint: disable=E0611
from tensorflow.contrib.image.python.ops.dense_image_warp \
    import _interpolate_bilinear
from template_updating import TemplateHistory


class SIRGraph:
    """ graph class containing tensorflow graph and access methods """

    def __init__(self, sir_options, video_options, template_options):
        self.sir_options = sir_options
        self.video_options = video_options
        self.template_options = template_options
        self.build_graph()

    def build_graph(self):
        """ build sir filter tensorflow graph, returns a dictionary
            containing necessary contact points for sir filter algorithms """

        self.graph = tf.Graph()
        self.graph.seed = self.sir_options['seed']
        # pylint: disable=E1129
        with self.graph.as_default():
            with tf.variable_scope("sir"):

                self.frame = tf.get_variable(
                    "frame",
                    [
                        self.video_options['height'],
                        self.video_options['width']],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)

                self.frame_input = tf.placeholder(
                    dtype=tf.float32,
                    shape=[self.video_options['height'],
                           self.video_options['width']])
                self.set_frame = self.frame.assign(self.frame_input)

                self.template = tf.get_variable(
                    'template',
                    [
                        self.template_options['height'],
                        self.template_options['width']],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)

                with tf.variable_scope("grid_coordinates"):
                    rspace = tf.linspace(
                        np.float32(-self.template_options['height']/2.0),
                        np.float32(self.template_options['height']/2.0),
                        self.template_options['height'], name='rspace')

                    cspace = tf.linspace(
                        np.float32(-self.template_options['width']/2.0),
                        np.float32(self.template_options['width']/2.0),
                        self.template_options['width'], name='cspace')

                    tgrid = tf.meshgrid(rspace, cspace, indexing='ij')
                    tgrid = tf.stack(tgrid, axis=2, name='tgrid')

                with tf.variable_scope("roi"):
                    self.roi_x = tf.get_variable(
                        "roi_X",
                        [4],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer)

                    # build magnification and rotation transformation
                    roi_c = tf.cos(self.roi_x[3])
                    roi_s = tf.sin(self.roi_x[3])
                    roi_rot = [[roi_c, -roi_s], [roi_s, roi_c]]
                    roi_rot_mag = roi_rot*self.roi_x[2]

                    # perform transformation contraction
                    roi_grid = tf.einsum('mnz,zk->mnk', tgrid, roi_rot_mag)

                    # translate
                    roi_grid = roi_grid + self.roi_x[0:2]

                    # shape tensors for _interpolate_bilinear batch
                    roi_grid = tf.reshape(
                        roi_grid,
                        [1,
                         (self.template_options['height'] *
                          self.template_options['width']),
                         2])
                    roi_frame = tf.reshape(
                        self.frame,
                        [1,
                         self.video_options['height'],
                         self.video_options['width'],
                         1])

                    # bilinear interpolate and reshape
                    roi_out = tf.squeeze(
                        _interpolate_bilinear(roi_frame, roi_grid))
                    roi_out = tf.reshape(
                        roi_out,
                        [self.template_options['height'],
                         self.template_options['width']],
                        name='roi_out')

                # system dynamics
                self.system_a = tf.constant(
                    [[1, 0, 1, 0, 0, 0],
                     [0, 1, 0, 1, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]], dtype=tf.float32)

                self.system_u = tf.constant(
                    [0, 0, 2, 2, 0.05, 0.02],
                    shape=(6, 1), dtype=tf.float32)

                self.sir_p = tf.get_variable(
                    "P",
                    [self.sir_options['particle_count'], 6],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)

                self.p_aux = tf.get_variable(
                    "P_aux",
                    [self.sir_options['particle_count'], 6],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer)

                self.p_seed = tf.placeholder(
                    dtype=tf.float32,
                    shape=self.sir_p.shape)
                self.seed_p = self.sir_p.assign(self.p_seed)

                sir_w = tf.get_variable("W",
                                        [self.sir_options['particle_count']],
                                        dtype=tf.float32,
                                        initializer=tf.ones_initializer)

                self.reset_w = sir_w.assign(
                    tf.fill(
                        tf.shape(sir_w),
                        1/self.sir_options['particle_count']))

                self.noise_p = tf.multiply(
                    tf.random_normal(self.sir_p.shape),
                    tf.reshape(self.system_u, [1, 6]), name='noise')
                prediction_p = tf.matmul(
                    self.sir_p,
                    self.system_a,
                    transpose_b=True) + self.noise_p

                self.predict_from_p = self.sir_p.assign(prediction_p)

                with tf.variable_scope("transform"):

                    # build magnification and rotation transformations
                    transform_c = tf.cos(self.sir_p[:, 5])
                    transform_s = tf.sin(self.sir_p[:, 5])
                    transform_rot = tf.stack(
                        [tf.stack([transform_c, -transform_s], axis=1),
                         tf.stack([transform_s, transform_c], axis=1)], axis=2)
                    transform_rot_mag = transform_rot * \
                        tf.reshape(
                            self.sir_p[:, 4],
                            [self.sir_options['particle_count'], 1, 1])

                    # perform transformation contraction
                    transform_grid = tf.einsum(
                        'mnz,pzk->pmnk', tgrid, transform_rot_mag)

                    # translate
                    transform_grid = transform_grid + \
                        tf.reshape(
                            self.sir_p[:, 0:2],
                            [self.sir_options['particle_count'], 1, 1, 2])

                with tf.variable_scope("interpolations"):
                    score_grid = tf.reshape(transform_grid, [1, -1, 2])
                    score_frame = tf.reshape(
                        self.frame,
                        [1,
                         self.video_options['height'],
                         self.video_options['width'],
                         1])
                    interpolations = _interpolate_bilinear(
                        score_frame, score_grid)
                    interpolations = tf.reshape(
                        interpolations,
                        [self.sir_options['particle_count'],
                         self.template_options['height'],
                         self.template_options['width']])

                with tf.variable_scope("Score"):
                    template_mean = tf.reduce_mean(
                        self.template, axis=[0, 1], keepdims=True)
                    mean_shifted_template = self.template-template_mean
                    e_template = tf.einsum(
                        'mn,mn->',
                        mean_shifted_template,
                        mean_shifted_template)

                    ss_mean = tf.reduce_mean(
                        interpolations,
                        axis=[1, 2],
                        keepdims=True)
                    mean_shifted_ss = interpolations-ss_mean
                    e_ss = tf.einsum(
                        'pmn,pmn->p', mean_shifted_ss, mean_shifted_ss)

                    # spatial-support and template products
                    e_template_ss = tf.einsum(
                        'pmn,mn->p', mean_shifted_ss, mean_shifted_template)

                    def score_init_time_0():
                        # template energy
                        e0_template = tf.get_variable(
                            "e0_template",
                            [],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        store_e0_template = e0_template.assign(e_template)
                        # spatial-support energy
                        e0_ss = tf.get_variable(
                            "e0_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        store_e0_ss = e0_ss.assign(e_ss)
                        # template-spatial support cross energy
                        e0_template_ss = tf.get_variable(
                            "e0_template_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        store_e0_template_ss = \
                            e0_template_ss.assign(e_template_ss)
                        return e0_template, store_e0_template, e0_ss, \
                            store_e0_ss, e0_template_ss, \
                            store_e0_template_ss

                    def score_init_time_1():
                        # template
                        e1_template = tf.get_variable(
                            "e1_template",
                            [],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e1_template = e1_template.assign(e_template)
                        # spatial-support energy
                        e1_ss = tf.get_variable(
                            "e1_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e1_ss = e1_ss.assign(e1_ss)
                        # template-spatial support cross energy
                        e1_template_ss = tf.get_variable(
                            "e1_template_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e1_template_ss = e1_template_ss.assign(
                            e1_template_ss)
                        return e1_template, shift_e1_template, e1_ss, \
                            shift_e1_ss, e1_template_ss, shift_e1_template_ss

                    def score_init_time_2():
                        # template energy
                        e2_template = tf.get_variable(
                            "e2_template",
                            [],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e2_template = e2_template.assign(e1_template)
                        # spatial-support energy
                        e2_ss = tf.get_variable(
                            "e2_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e2_ss = e2_ss.assign(e1_ss)
                        # template-spatial support cross energy
                        e2_template_ss = tf.get_variable(
                            "e2_template_ss",
                            [self.sir_options['particle_count']],
                            dtype=tf.float32,
                            initializer=tf.ones_initializer)
                        shift_e2_template_ss = e2_template_ss.assign(
                            e1_template_ss)
                        return e2_template, shift_e2_template, e2_ss, \
                            shift_e2_ss, e2_template_ss, shift_e2_template_ss

                    if self.sir_options['score_type'] == 'NCC':
                        e0_template, \
                            self.store_e0_template, \
                            e0_ss, \
                            self.store_e0_ss, \
                            e0_template_ss, \
                            self.store_e0_template_ss = \
                            score_init_time_0()

                        corr = (e0_template_ss /
                                (tf.sqrt(e0_template)*tf.sqrt(e0_ss)))

                    elif self.sir_options['score_type'] == 'ASV':
                        e0_template, \
                            self.store_e0_template, \
                            e0_ss, \
                            self.store_e0_ss, \
                            e0_template_ss, \
                            self.store_e0_template_ss = \
                            score_init_time_0()
                        e1_template, \
                            self.shift_e1_template, \
                            e1_ss, self.shift_e1_ss, \
                            e1_template_ss, \
                            self.shift_e1_template_ss = \
                            score_init_time_1()

                        corr = ((e0_template_ss+e1_template_ss) /
                                (tf.sqrt(e0_template+e1_template) *
                                 tf.sqrt(e0_ss+e1_ss)))

                    elif self.sir_options['score_type'] == 'ASVHO':
                        e0_template, \
                            self.store_e0_template, \
                            e0_ss, \
                            self.store_e0_ss, \
                            e0_template_ss, \
                            self.store_e0_template_ss = \
                            score_init_time_0()
                        e1_template, \
                            self.shift_e1_template, \
                            e1_ss, self.shift_e1_ss, \
                            e1_template_ss, \
                            self.shift_e1_template_ss = \
                            score_init_time_1()
                        e2_template, \
                            self.shift_e2_template, \
                            e2_ss, \
                            self.shift_e2_ss, \
                            e2_template_ss, \
                            self.shift_e2_template_ss = \
                            score_init_time_2()

                        corr = ((e0_template_ss +
                                 e1_template_ss +
                                 e2_template_ss) /
                                (tf.sqrt(
                                    e0_template +
                                    e1_template +
                                    e2_template) *
                                 tf.sqrt(
                                     e0_ss +
                                     e1_ss +
                                     e2_ss)))

                    score = tf.exp(-100*(1.0-corr))
                    score = score / (tf.reduce_sum(score))
                    score_out = tf.get_variable(
                        "score_out",
                        [self.sir_options['particle_count']],
                        dtype=tf.float32,
                        initializer=tf.ones_initializer)
                    self.store_score = score_out.assign(score)

                with tf.variable_scope("resampling"):
                    logprobs_w = tf.reshape(tf.log(sir_w), [1, -1])
                    ridx_w = tf.squeeze(
                        tf.multinomial(
                            logprobs_w,
                            num_samples=self.sir_options['particle_count']))

                    resample_indices = tf.get_variable(
                        "resample_indices",
                        [self.sir_options['particle_count']],
                        dtype=tf.int64,
                        initializer=tf.zeros_initializer)
                    self.store_ridx = resample_indices.assign(ridx_w)

                    r_p = tf.gather(self.sir_p, resample_indices)
                    self.resample_p = self.sir_p.assign(r_p)
                    if self.sir_options['score_type'] == 'ASV':
                        self.resample_e1_template_ss = e1_template_ss.assign(
                            tf.gather(e1_template_ss, resample_indices))
                        self.resample_e1_ss = e1_ss.assign(
                            tf.gather(e1_ss, resample_indices))
                    if self.sir_options['score_type'] == 'ASVHO':
                        self.resample_e1_template_ss = e1_template_ss.assign(
                            tf.gather(e1_template_ss, resample_indices))
                        self.resample_e1_ss = e1_ss.assign(
                            tf.gather(e1_ss, resample_indices))
                        self.resample_e2_template_ss = e2_template_ss.assign(
                            tf.gather(e2_template_ss, resample_indices))
                        self.resample_e2_ss = e2_ss.assign(
                            tf.gather(e2_ss, resample_indices))

                self.w_update = score_out * sir_w
                self.w_update = (self.w_update /
                                 tf.reduce_sum(self.w_update, axis=0))
                self.update_w = sir_w.assign(self.w_update)

                self.estimate = tf.reduce_sum(
                    tf.reshape(
                        self.w_update,
                        [-1, 1])*self.sir_p, axis=0, name='estimate')

                with tf.variable_scope('template_update'):

                    self.set_roi_to_template = self.template.assign(roi_out)

                    # establish best template history if not using estimate
                    if 'ESTIMATE' not in self.sir_options['update_method']:
                        if 'WEIGHT' in self.sir_options['update_method']:
                            max_source = sir_w
                        elif 'SCORE' in self.sir_options['update_method']:
                            max_source = score
                        elif 'CORRELATION' in \
                             self.sir_options['update_method']:
                            max_source = corr

                        # max among current particles
                        max_idx = tf.argmax(max_source)
                        best_current_template = interpolations[max_idx]
                        best_current_value = max_source[max_idx]
                        self.template_history = TemplateHistory(
                            self.graph,
                            self.sir_options['historical_length'],
                            best_current_template,
                            best_current_value
                        )

                        if 'SVD' in self.sir_options['update_method']:
                            best_historical_template = \
                                self.template_history.get_svd()
                        else:
                            best_historical_template = \
                                self.template_history.get_best()

                        self.update_from_best = self.template.assign(
                            best_historical_template)

                # useful ops
                self.store_aux_p = self.p_aux.assign(self.sir_p)
                self.restore_p_from_aux = self.sir_p.assign(self.p_aux)
                # number of effective particles calculation
                self.neff = 1.0/tf.einsum('p,p->', sir_w, sir_w)

    def set_template_roi(self, session, row, col, mag, rot):
        est_feed = {self.roi_x: [row, col, mag, rot]}
        session.run(self.set_roi_to_template, feed_dict=est_feed)

    def maintain_template(self, session, do_update):
        if 'ESTIMATE' not in self.sir_options['update_method']:
            self.template_history.push_template(session)

        if not do_update:
            return

        if 'ESTIMATE' in self.sir_options['update_method']:
            est = session.run(self.estimate)
            self.set_template_roi(session, est[0], est[1], est[4], est[5])
        else:
            session.run(self.update_from_best)

    def store_energies(self, session):
        session.run(self.store_e0_template_ss)
        session.run(self.store_e0_template)
        session.run(self.store_e0_ss)

    def shift_energies(self, session):
        if self.sir_options['score_type'] == 'NCC':
            pass
        elif self.sir_options['score_type'] == 'ASV':
            session.run(self.shift_e1_template_ss)
            session.run(self.shift_e1_template)
            session.run(self.shift_e1_ss)
        if self.sir_options['score_type'] == 'ASVHO':
            session.run(self.shift_e1_template_ss)
            session.run(self.shift_e1_template)
            session.run(self.shift_e1_ss)
            session.run(self.shift_e2_template_ss)
            session.run(self.shift_e2_template)
            session.run(self.shift_e2_ss)
