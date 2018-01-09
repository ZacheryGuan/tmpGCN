from __future__ import division
import time
from copy import deepcopy
from os import cpu_count
from gcn.utils import preprocess_model_config
import argparse
import pprint
# import math
# Model 11: nearest weighted
# Model 12: see draft
# Model 13: Model 9 + Model 11
# Model 14: variant of weighted Model 11
# Model 15: Model 9 + Model 14
# Model 16: use gcn to extend train set
#           'Model_to_add_label':   the model used to extend train set
#           'Model_to_predict'  :   the final model
#           't' :   if t is a scalar, it's the total number of additional labels
#                   if t is a list, it's the number of additional labels of each class
# 17 label propagation
# 18 self learning with label propagation
# Model 19: gcn and lp
#           'Model19' : 'union',        # 'union' | 'intersection'
# 22 LP with features
# validate: (True | False) Whether use validation set
# 23 svm
# 24 Test21
# 25 Test22
# 26 Model9+modify adjacency matrix
configuration ={
    # repeating times
    'repeating'             : 1,

    # The default model configuration
    'default':{
        'dataset'           : 'cora',     # 'Dataset string. (cora | citeseer | pubmed | CIFAR-Fea | USPS-Fea)'
        'train_size'        : 1,         # if train_size is a number, then use TRAIN_SIZE%% labels.
        # 'train_size'        : [20, 20, 20, 20, 20, 20, 20], # if train_size is a list of numbers, then it specifies training lables for each class.
        'validation_size'   : 20,           # 'Use VALIDATION_SIZE% data to train model'
        'validate'          : False,        # Whether use validation set
        'conv'              : 'gcn',        # 'conv type. (gcn | cheby | chebytheta | gcn_rw | taubin)'
        'max_degree'        : 2,            # 'Maximum Chebyshev polynomial degree.'
        'learning_rate'     : 0.02,         # 'Initial learning rate.'
        'epochs'            : 200,          # 'Number of epochs to train.'

        # config the absorption probability
        'Model'             : 0,
        # 's'                 : 100,
        # 's' in the construction of  absorption probability
        # if s = -1 and Model == 1, non zero elements in each row equals to the original adjacency matrix
        'absorption_type'   : 'binary',     # When Model == 1, the new constructed adjacency matrix is either 'binary' or 'weighted'
        'mu'                : 1,          # 'mu' in the Model5
        't'                 : 500,           # In model9, top 't' nodes will be reserved.
        't2'                : 100,          # In model 21
        'lambda'            : 0,
        'Model11'           : 'nearest',     # 'weighted' | 'nearest'
        'k'                 : -1,           # k in model 11. if k<0, then it is determined by program.
        'Model_to_add_label': { 'Model' :0 }, # for model 16
        'Model_to_predict'  : { 'Model' :0 }, # for model 16
        'Model19'           : 'union',        # 'union' | 'intersection'
        'classifier'        : 'svm',            # 'svm' | 'tree'
        'svm_kernel'        : 'rbf',        # 'rbf' | 'poly' | 'rbf' | 'sigmoid', model 23
        'gamma'             : 1e-5,         # gamma for svm, see scikit-learn document, model 23
        'svm_degree'        : 4,
        'tree_depth'        : None,

        'connection'        : 'cc',
        # A string contains only char "c" or "f" or "d".
        # "c" stands for convolution.
        # "f" stands for fully connected.
        # "d" stands for dense net.
        # See layer_size for details.

        'layer_size'        : [16],
        # A list or any sequential object. Describe the size of each layer.
        # e.g. "--connection ccd --layer_size [7,8]"
        #     This combination describe a network as follow:
        #     input_layer --convolution-> 7 nodes --convolution-> 8 nodes --dense-> output_layer
        #     (or say: input_layer -c-> 7 -c-> 8 -d-> output_layer)

        'dropout'           : 0.5,          # 'Dropout rate (1 - keep probability).'
        'weight_decay'      : 5e-4,         # 'Weight for L2 loss on embedding matrix.'

        'early_stopping'    : 0,
        # 'Tolerance for early stopping (# of epochs).
        # Non positive value means never early stop.'

        'random_seed'       : int(time.time()),     #'Random seed.'
        'feature'           : 'bow',        # 'bow' | 'tfidf' | 'none'.

        'smoothing'         : None,        # 'poly'| 'ap'  | 'taubin' | 'test21' | None
        'alpha'             : 1e-6,         # 'alpha' in the construction of  absorption probability
        'beta'              : 0.05,
        'poly_parameters'   : [1,-2,1],           # coefficients of p(L_rw)
        'taubin_lambda'     : 0.3,
        'taubin_mu'         : -0.31,
        'taubin_repeat'     : 5,
        'taubin_f'          : 0.7,
        'taubin_t'          : 0.2,

        'logging'           : False,         # 'Weather or not to record log'
        'logdir'            : '',           # 'Log directory.''
        'name'              : '',           # 'name of the model. Serve as an ID of model.'
        # if logdir or name are empty string,
        # logdir or name will be auto generated according to the
        # structure of the network

        'threads'           : 2*cpu_count(),  #'Number of threads'
        'train'             : True,
        'drop_inter_class_edge': False,
        'loss_func'         :'default',     #'imbalance'
        'ws_beta'           : 20,
        
        
        'max_triplet':1000  #for cora
    },

    # The list of model to be train.
    # Only configurations that's different with default are specified here
    'model_list':
    [
        {
            'Model' : 0,
            'connection'        : 'cc',
            'conv'              : 'gcn',
            'hard' : False,
            'feature_normalize':False            
        }
    ] +
    [
        {
            'Model' : 0,
            'connection'        : 'cc',
            'conv'              : 'gcn',
            'loss_func': 'triplet',
            'MARGIN' : margin,
            'triplet_lamda':triplet_lamda,
            'hard' : False,
            'feature_normalize':True
        }  for margin in [0.3, 0.8, 2.0]
        for triplet_lamda in [1, 5, 10]
        
    ]+
    [
        {
            'Model' : 0,
            'connection'        : 'cc',
            'conv'              : 'gcn',
            'loss_func': 'triplet',
            'MARGIN' : margin,
            'triplet_lamda':triplet_lamda,
            'hard' : False,
            'feature_normalize':False,
        } for margin in [0.3, 0.8, 2.0]
        for triplet_lamda in [1, 5, 10]
    ]
#     
#     +
#     [
#         {
#             'Model' : 0,
#             'connection'        : 'cc',
#             'conv'              : 'gcn',
#             'loss_func': 'triplet',
#             'MARGIN' : margin,
#             'triplet_lamda':1.0,
#             'hard': True
           
#        } for margin in [0.1,0.2, 0.3, 0.4, 0.5]
        
#     ]
#     
   
    
}

# Parse args
parser = argparse.ArgumentParser(description=(
    "This is used to train and test Graph Convolution Network for node classification problem.\n"
    "Most configuration are specified in config.py, please read it and modify it as you want."))
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--dataset", type=str)
parser.add_argument("--train_size", type=float)
parser.add_argument("--repeating", type=int)
parser.add_argument("--validate", type=bool, help='0 | 1')
parser.add_argument("--loss_func", type=str)
parser.add_argument("--ws_beta", type=int)

args = parser.parse_args()
print(args)
if args.dataset is not None:
    configuration['default']['dataset'] = args.dataset
if args.train_size is not None:
    configuration['default']['train_size'] = args.train_size
if args.repeating is not None:
    configuration['repeating']=args.repeating
if args.validate is not None:
    configuration['default']['validate']=args.validate
if args.loss_func is not None:
    configuration['default']['loss_func']=args.loss_func
if args.ws_beta is not None:
    configuration['default']['ws_beta']=args.ws_beta
pprint.PrettyPrinter(indent=4).pprint(configuration)
# exit()

def set_default_attr(model):
    model_config = deepcopy(configuration['default'])
    model_config.update(model)
    return model_config

configuration['model_list'] = list(map(set_default_attr,
    configuration['model_list']))

for model_config in configuration['model_list']:
    preprocess_model_config(model_config)
