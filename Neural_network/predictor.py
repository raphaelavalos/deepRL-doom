import tensorflow as tf
from subnets import perception_net, measurement_net, goal_net, expectation_net, action_net



class DOOM_Predictor():
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    def __init__(self, conf, parameters):
        self._visual_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(-1, conf.arch['vis_size'], conf.arch['vis_size'], 1),
                                                  name='visual_input')
        self._measurement_placeholder = tf.placeholder(dtype=tf.float32,
                                                       shape=(-1, conf.arch['meas_size']),
                                                       name='meas_input')
        self._goal_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(-1, conf.arch['num_time'], conf.arch['goal_size']))
        """
        parameters keys are : 'perception','measurement','goal','expectation','action'
        each value is a dict itself containing keys : 'filters','kernel_size','strides','fc_out'
        each value of this dict are following:
            filters : list of integers corresponding at the output dim of the respective convolutions
            kernel_size : list of [kernel_width, kernel_width]
            strides : list of [strides_width, strides_width]
            fc_out : list of output dimensions for fully connected layers    
        """
        self.parameters = parameters



    def predict(self, data):
        # data is (s=Tensor[batch_size,image_width,image_height], m=Tensor[number_measurement], g=Tensor[horizon])
        images , measures, goals = data
        perception_out = perception_net(images, self.parameters.get('perception'))
        measurement_out = measurement_net(measures, self.parameters.get('measurement'))
        goal_out =  goal_net(goals, self.parameters.get('goals'))
        expectation_out = expectation_net(perception_out,
                                          measurement_out,
                                          goal_out,
                                          self.parameters.get('expectation'))
        action_out = action_net(perception_out,
                                          measurement_out,
                                          goal_out,
                                          self.parameters.get('action'))

        return expectation_out, action_out


    def optimize(self, data):
        # data is ((batch,s=image,m=measurement,g=goal), (batch, target))
        pass

    def chose_action(self, data):
        # data is (batch,s=image,m=measurement,g=goal)
        # calls predict
        pass


