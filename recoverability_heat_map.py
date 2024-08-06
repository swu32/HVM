from associative_learning_human_experiment import *
from HVM_human_experiment import *
from HCM_human_experiment import *
# theta = [0.996, 0.99, 0.96, 0.9]
n_sample = 1 # let's say 10 for now, too many then there is not enough compute time
theta = 0.96

general_path = '/Users/swu/Documents/MouseHCM/recoverability_analysis'
name_hvm = 'simulation_data_model_hvm'
name_al = 'simulation_data_model_al'
name_hcm = 'simulation_data_model_hcm'


def simulation_generator(modeltype, n_sample, theta = 0.996):
    general_path = '/Users/swu/Documents/MouseHCM/recoverability_analysis'
    name_hvm = 'simulation_data_model_hvm'
    name_al = 'simulation_data_model_al'
    name_hcm = 'simulation_data_model_hcm'

    savepath = ''
    if modeltype == 'AL':
        savepath = general_path + '/' + name_al + 'theta=' + str(theta) + 'sample='+str(n_sample)+ '.csv'
        AL_experiment2(theta = theta, save_path = savepath)
    elif modeltype == 'HCM':
        savepath = general_path + '/' + name_hcm + 'theta=' + str(theta) + 'sample='+str(n_sample)+ '.csv'
        HCM_experiment2(theta = theta, save_path = savepath)
    elif modeltype == 'HVM':
        savepath = general_path + '/' + name_hvm + 'theta=' + str(theta) + 'sample='+str(n_sample)+ '.csv'
        HVM_experiment2(theta = theta, save_path = savepath)

    return

# generate different models with the same forgetting parameter, mark the files by the sample number and the paramter number

# load 10 simulated model of each type and compute the R square of the simulated model with the other

modeltype = ['AL','HCM','HVM']
for i in range(0, n_sample):
    for model in modeltype:
        simulation_generator(model, i, theta = theta)

print('model simulation is done')


