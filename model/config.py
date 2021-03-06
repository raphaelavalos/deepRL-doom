# DOOM_Predictor_v2 configuration dictionary


def build_conf(args):
    conf = {}

    if args.mode in [1, 2]:
        conf['action_nbr'] = 5
        conf['offsets_dim'] = 6
        conf['measurement_dim'] = 1

    elif args.mode in [3, 4]:
        conf['action_nbr'] = 256
        conf['offsets_dim'] = 6
        conf['measurement_dim'] = 3

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
    conf['expectation']['dense_1']['units'] = conf['offsets_dim'] * conf['measurement_dim']
    conf['action'] = {}
    conf['action']['action_nbr'] = conf['action_nbr']
    conf['action']['offsets_dim'] = 6
    conf['action']['measurement_dim'] = conf['measurement_dim']
    conf['action']['offsets_dim'] = conf['offsets_dim']
    conf['action']['dense'] = {}
    conf['action']['dense']['dense_nbr'] = 2
    conf['action']['dense']['dense_0'] = {}
    conf['action']['dense']['dense_0']['units'] = 512
    conf['action']['dense']['dense_1'] = {}
    conf['action']['dense']['dense_1']['units'] = conf['offsets_dim'] * conf['measurement_dim'] * conf['action_nbr']

    conf['optimizer'] = {}
    conf['optimizer']['learning_rate'] = args.learning_rate
    conf['optimizer']['decay_steps'] = args.decay_steps
    conf['optimizer']['decay_rate'] = args.decay_rate

    conf['mode_path'] = args.mode_path
    conf['nbr_of_simulators'] = args.nbr_of_simulators
    conf['memory'] = {}
    conf['memory']['capacity'] = args.memory
    conf['device'] = args.device
    conf['image_resolution'] = (84, 84, 1)
    conf['offsets'] = [1, 2, 4, 8, 16, 32]
    conf['skip_tic'] = args.skip_tic
    conf['experiment_name'] = args.name
    conf['use_goal'] = False
    conf['restore_path'] = args.restore_path
    return conf
