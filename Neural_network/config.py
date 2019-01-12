
""" PERCEPTION params """
s_filters = [32,64,64]
s_kernel_size = [[8,8],[4,4],[3,3]]
s_strides = [[4,4],[2,2],[1,1]]
s_fc_out = [512]

""" MEASURMENT params """
m_filters = []
m_kernel_size = []
m_strides = []
m_fc_out = [128,128,128]

""" GOAL params """
g_filters = []
g_kernel_size = []
g_strides = []
g_fc_out = [128,128,128]

""" EXPECTATION params """
e_filters = []
e_kernel_size = []
e_strides = []
e_fc_out = [512]

""" ACTION params """
a_filters = []
a_kernel_size = []
a_strides = []
a_fc_out = [512]

""" ARCH params """
vis_size = 84
meas_size = 3
num_time = 6
num_action = 256

def build_parameters():
    parameters = {'perception' : {},
                  'measurement' : {},
                  'goal' : {},
                  'expectation' : {},
                  'action' : {},
                  'arch' :  {}}
    
    #Build of perception dict
    
    parameters['perception'] = {'filters': s_filters,
                                'kernel_size': s_kernel_size,
                                'strides' : s_strides,
                                'fc_out' : s_fc_out}
    parameters['measurement'] = {'filters': m_filters,
                                'kernel_size': m_kernel_size,
                                'strides' : m_strides,
                                'fc_out' : m_fc_out}
    parameters['goal'] = {'filters': g_filters,
                                'kernel_size': g_kernel_size,
                                'strides' : g_strides,
                                'fc_out' : g_fc_out}
    parameters['expectation'] = {'filters': e_filters,
                                'kernel_size': e_kernel_size,
                                'strides' : e_strides,
                                'fc_out' : e_fc_out}
    parameters['action'] = {'filters': a_filters,
                                'kernel_size': a_kernel_size,
                                'strides' : a_strides,
                                'fc_out' : a_fc_out}
    parameters['arch'] = {'vis_size' : vis_size,
                          'meas_size' : meas_size,
                          'num_time' : num_time,
                          'num_action' : num_action}
    return parameters
    
    