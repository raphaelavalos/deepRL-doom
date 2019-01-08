import tensorflow as tf


class DOOM_Predictor:
    """
    DOOM Predictor implenting the paper 'Learning To Act By Predicting The Future'
    """

    def __init__(self, conf):
        pass

    def predict(self, data):
        # data is (batch,s=image,m=measurement,g=goal)
        pass

    def optimize(self, data):
        # data is ((batch,s=image,m=measurement,g=goal), (batch, target))
        pass

    def chose_action(self, data):
        # data is (batch,s=image,m=measurement,g=goal)
        # calls predict
        pass


