# DOOM_Predictor_v2 configuration dictionary

conf = {}

conf['perception'] = {}
conf['perception']['conv_nbr'] = 3  # TODO: Can infer this value

conf['perception']['conv_0'] = {}
conf['perception']['conv_0']['filters'] = 32
conf['perception']['conv_0']['kernel_size'] = 8
conf['perception']['conv_0']['stride'] = 4

conf['perception']['conv_1'] = {}
conf['perception']['conv_1']['filters'] = 64
conf['perception']['conv_1']['kernel_size'] = 4
conf['perception']['conv_1']['stride'] = 2

conf['perception']['conv_2'] = {}
conf['perception']['conv_2']['filters'] = 64
conf['perception']['conv_2']['kernel_size'] = 3
conf['perception']['conv_2']['stride'] = 1

conf['perception']['dense'] = {}
conf['perception']['dense']['units'] = 512

conf['measurement'] = {}
conf['measurement']['dense_nbr'] = 3
conf['measurement']['dense_0'] = {}
conf['measurement']['dense_0']['units'] = 128
conf['measurement']['dense_1'] = {}
conf['measurement']['dense_1']['units'] = 128
conf['measurement']['dense_2'] = {}
conf['measurement']['dense_2']['units'] = 128

conf['goal'] = {}
conf['goal']['dense_nbr'] = 3
conf['goal']['dense_0'] = {}
conf['goal']['dense_0']['units'] = 128
conf['goal']['dense_1'] = {}
conf['goal']['dense_1']['units'] = 128
conf['goal']['dense_2'] = {}
conf['goal']['dense_2']['units'] = 128

conf['expectation'] = {}
conf['expectation']['dense_nbr'] = 2
conf['expectation']['dense_0'] = {}
conf['expectation']['dense_0']['units'] = 512
conf['expectation']['dense_1'] = {}
conf['expectation']['dense_1']['units'] = 3*6

conf['action'] = {}
conf['action']['action_nbr'] = 256
conf['action']['offsets_dim'] = 6
conf['action']['dense'] = {}
conf['action']['dense']['dense_nbr'] = 2
conf['action']['dense']['dense_0'] = {}
conf['action']['dense']['dense_0']['units'] = 512
conf['action']['dense']['dense_1'] = {}
conf['action']['dense']['dense_1']['units'] = 3*6*256

conf['optimizer'] = {}
conf['optimizer']['learning_rate'] = 0.001
conf['optimizer']['decay_steps'] = 10000  # TODO: change those values
conf['optimizer']['decay_rate'] = 0.96

conf['choose_action'] = {}
conf['choose_action']['action_nbr'] = 256
conf['offsets_dim'] = 6
conf['measurement_dim'] = 3

