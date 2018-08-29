""" class and methods for template updating """
import tensorflow as tf


class TemplateHistory():
    """ container for template history """

    def __init__(self, graph, history_length,
                 template_source, value_source):
        self.graph = graph
        self.history_length = history_length
        self._template_source = template_source
        self._t_h = template_source.shape[0]
        self._t_w = template_source.shape[1]
        self._value_source = value_source
        self._seeded_count = history_length-1
        self.build_graph()

    def build_graph(self):
        """ build template history graph """
        with self.graph.as_default():
            with tf.variable_scope('template_history'):
                self.template_history = tf.get_variable(
                    'template_history',
                    [self.history_length, self._t_h, self._t_w],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer
                )
                self.value_history = tf.get_variable(
                    'value_history',
                    [self.history_length],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer
                )
                self._push_template_assign = self.template_history.assign(
                    tf.concat(
                        [tf.expand_dims(self._template_source, 0),
                         self.template_history[0:-1]],
                        axis=0
                    )
                )
                self._push_value_assign = self.value_history.assign(
                    tf.concat(
                        [tf.expand_dims(self._value_source, 0),
                         self.value_history[0:-1]],
                        axis=0
                    )
                )
                # best template retrieval graph
                self._max_index = tf.argmax(self.value_history)
                self._best_template = self.template_history[self._max_index]
                self._best_value = self.value_history[self._max_index]

                # SVD generated template graph
                # reshape into rows and transpose so templates are in columns
                spaghettified = tf.reshape(
                    self.template_history, [self.history_length, -1])
                spaghettified = tf.transpose(spaghettified)
                s, u, v = tf.svd(spaghettified)  # pylint: disable=C0103
                # pylint: disable=C0103
                self._SVD_composite = u[:, 0] * s[0] * v[0, 0]
                self._SVD_composite = tf.reshape(
                    self._SVD_composite, [self._t_h, self._t_w])

    def push_template(self, session):
        """ push template and value into history queue """

        while True:
            session.run(self._push_template_assign)
            session.run(self._push_value_assign)
            if self._seeded_count == 0:
                break
            else:
                self._seeded_count -= 1

    def get_best(self):
        """ retrieve best template tensor given associated value """
        return self._best_template

    def get_svd(self):
        """ retrieve SVD generated template tensor """
        return self._SVD_composite
