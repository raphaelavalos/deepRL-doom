import tensorflow as tf
import numpy as np
from .subnets import perception_net, measurement_net, goal_net, expectation_net, action_net
from .config import build_parameters


class DOOM_Predictor():
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    # TODO : add the goal vector as parameter

    def __init__(self, parameters=build_parameters()):
        self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, parameters.get('arch')['vis_size'],
                                                         parameters.get('arch')['vis_size'], 1),
                                                  name='visual_input')
        self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(None, parameters.get('arch')['meas_size']),
                                                       name='meas_input')
        self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None,
                                                       parameters.get('arch')['num_time'] * parameters.get('arch')[
                                                           'meas_size']))
        self._true_action_placeholder = tf.placeholder(dtype=tf.int32,
                                                       shape=(None,))
        self._true_future_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(None, parameters.get('arch')['meas_size'] *
                                                              parameters.get('arch')['num_time']),
                                                       name='ground_truth')

        """
        parameters keys are : 'arch','perception','measurement','goal','expectation','action'
        each value except 'arch' is a dict itself containing keys : 'filters','kernel_size','strides','fc_out'
        each value of this dict are following:
            filters : list of integers corresponding at the output dim of the respective convolutions
            kernel_size : list of [kernel_width, kernel_width]
            strides : list of [strides_width, strides_width]
            fc_out : list of output dimensions for fully connected layers  
            
        arch dict contain what's relative to the inputs dimensions:
            vis_size : image input
            meas_size : number of measures
            num_time : number of horizon at which we look
            num_action : number of possible actions
        """
        self.parameters = parameters

    def build_net(self):

        perception_out = perception_net(self._visual_placeholder, self.parameters.get('perception'))
        measurement_out = measurement_net(self._measurement_placeholder, self.parameters.get('measurement'))
        goal_out = goal_net(self._goal_placeholder, self.parameters.get('goal'))
        expectation_out = expectation_net(perception_out,
                                          measurement_out,
                                          goal_out,
                                          self.parameters.get('expectation'),
                                          self.parameters.get('arch'))

        action_out = tf.reshape(action_net(perception_out,
                                           measurement_out,
                                           goal_out,
                                           self.parameters.get('action'),
                                           self.parameters.get('arch')),
                                [-1, self.parameters.get('arch')['num_action'],
                                 self.parameters.get('arch')['meas_size'] * self.parameters.get('arch')['num_time']])

        self.output = tf.add(action_out, expectation_out)

    def chose_action(self, output, goal_vec, epsilon):
        # return the gready action choice  for any output
        p = np.random.random()
        if p > epsilon:
            return tf.argmax(tf.tensordot(output, goal_vec, axes=1), axis=1)
        else:
            return np.random.randint(self.parameters['arch']['num_action'])

    def optimize(self, data, epochs, step, batch_size, save_freq, lr=0.01, model_path='train/model'):

        # data is (s=array[batch_size,image_width,image_height], m=array[number_measurement], g=array[horizon])

        # TODO: Finish tf.gather to get right output
        with tf.name_scope('Loss'):
            self.loss = tf.losses.mean_squared_error(self._true_future_placeholder,
                                                     tf.gather_nd(self.output, tf.stack(((np.arange(batch_size)),
                                                                                         self._true_action_placeholder),
                                                                                        axis=1)))

        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            print("Training started")
            for epoch in range(epochs):
                avg_loss = 0.
                # total_batch = int(data['s'].shape[0]/batch_size)
                # Loop over all batches
                for i in range(step):
                    image, meas, goal, true_meas, true_action = data.get_batch(batch_size)
                    b_dict = {self._visual_placeholder: image,
                              self._measurement_placeholder: meas,
                              self._goal_placeholder: goal,
                              self._true_future_placeholder: true_meas,
                              self._true_action_placeholder: true_action}

                    _, l = sess.run([self.optimizer, self.loss],
                                    feed_dict=b_dict)
                    # Compute average loss
                    avg_loss += l / step

                    if (i % save_freq) == 0:
                        save_path = saver.save(sess, model_path + "epoch_%s_step%s.tf" % (epoch, i))
                        print("Model saved in file: %s" % save_path)

                print("Epoch: ", '%02d' % (epoch + 1), "  =====> Loss=", "{:.9f}".format(avg_loss))
