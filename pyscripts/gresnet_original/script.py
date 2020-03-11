from DatasetLoader import DatasetLoader
from MethodDeepGCNResNet import MethodDeepGCNResNet
from MethodDeepGATResNet import MethodDeepGATResNet
from MethodDeepLoopyResNet import MethodDeepLoopyResNet
from ResultSaving import ResultSaving
from SettingCV import SettingCV
from EvaluateAcc import EvaluateAcc
import numpy as np
import torch.nn as nn
import torch

#--------------------------------------------------------

#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'cora'

if dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500

#---- Deep GCN ResNet baseline method ----
if 0:
    for depth in [1, 2, 3, 4, 5, 6, 7]:
        for residual_type in ['naive', 'raw', 'graph_naive', 'graph_raw']:

            acc_test_list = []
            print('Method: GCN, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type)
            for iter in range(1):
                #---- parameter section -------------------------------
                lr = 0.01
                epoch = 1000
                weight_decay = 5e-4
                c = 0.1
                seed = iter
                dropout = 0.5
                nhid = 16
                #------------------------------------------------------

                #---- objection initialization setction ---------------
                print('Start')

                data_obj = DatasetLoader('', '')
                data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
                data_obj.c = c
                data_obj.method_type = 'GCN'

                method_obj = MethodDeepGCNResNet(nfeature, nhid, nclass, dropout, seed, depth)
                method_obj.lr = lr
                method_obj.epoch = epoch
                method_obj.residual_type = residual_type

                result_obj = ResultSaving('', '')
                result_obj.result_destination_folder_path = './result/GResNet/'
                result_obj.result_destination_file_name = 'DeepGCNResNet_' + dataset_name + '_' + residual_type + '_depth_' + str(depth) + '_iter_' + str(iter)

                setting_obj = SettingCV('', '')

                evaluate_obj = EvaluateAcc('', '')
                #------------------------------------------------------

                #---- running section ---------------------------------
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                acc_test = setting_obj.load_run_save_evaluate()
                print('Testing Acc: ', acc_test)
                acc_test_list.append(acc_test)
                print('*******************************')
                #------------------------------------------------------
            print(acc_test_list)
            print(np.mean(acc_test_list), np.std(acc_test_list))
            print('Finished')


#--------------------------------------------------------

#---- Deep Sparse GAT baseline method ----
if 0:
    for depth in [1, 2, 3, 4, 5, 6, 7]:
        for residual_type in ['naive', 'raw', 'graph_naive', 'graph_raw']:

            cuda_tag = False
            print('Method: GAT, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type)
            method_type = 'GAT'
            acc_test_list = []
            for iter in range(1):
                # ---- parameter section -------------------------------
                lr = 0.005
                epoch = 500
                weight_decay = 5e-4
                c = 0.1
                seed = iter
                dropout = 0.6
                nhid = 8
                nb_heads = 8
                alpha = 0.2
                patience = 100
                # ------------------------------------------------------

                # ---- objection initialization setction ---------------
                print('Start')

                data_obj = DatasetLoader('', '')
                data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
                data_obj.c = c
                data_obj.method_type = method_type

                method_obj = MethodDeepGATResNet(nfeature, nhid, nclass, dropout, seed, alpha, nb_heads, depth, cuda_tag)

                method_obj.lr = lr
                method_obj.epoch = epoch
                method_obj.weight_decay = weight_decay
                method_obj.residual_type = residual_type

                result_obj = ResultSaving('', '')
                result_obj.result_destination_folder_path = './result/GResNet/'
                result_obj.result_destination_file_name = 'DeepGATResNet_' + dataset_name + '_' + residual_type + '_depth_' + str(
                    depth) + '_iter_' + str(iter)

                setting_obj = SettingCV('', '')

                evaluate_obj = EvaluateAcc('', '')
                # ------------------------------------------------------

                # ---- running section ---------------------------------
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                acc_test = setting_obj.load_run_save_evaluate()
                print('Testing Acc: ', acc_test)
                acc_test_list.append(acc_test)
                print('*******************************')
                # ------------------------------------------------------
            print(acc_test_list)
            print(np.mean(acc_test_list), np.std(acc_test_list))
            print('Finished')


#--------------------------------------------------------

#---- Deep LoopyNet baseline method ----
if 0:
    for depth in [1, 2, 3, 4, 5, 6, 7]:
        for residual_type in ['naive', 'raw', 'graph_naive', 'graph_raw']:
            cuda_tag = False

            print('Method: LoopyNet, dataset: ' + dataset_name + ', depth: ' + str(depth) + ', residual type: ' + residual_type)
            acc_test_list = []
            for iter in range(1):
                #---- parameter section -------------------------------
                lr = 0.01
                epoch = 1000
                weight_decay = 5e-4
                c = 0.1
                seed = iter
                dropout = 0.5
                nhid = 16
                #------------------------------------------------------

                #---- objection initialization setction ---------------
                print('Start')

                data_obj = DatasetLoader('', '')
                data_obj.dataset_source_folder_path = './data/' + dataset_name + '/'
                data_obj.c = c
                data_obj.method_type = 'LoopyNet'

                method_obj = MethodDeepLoopyResNet(nfeature, nhid, nclass, dropout, seed, depth, cuda_tag)
                method_obj.lr = lr
                method_obj.epoch = epoch
                method_obj.residual_type = residual_type

                result_obj = ResultSaving('', '')
                result_obj.result_destination_folder_path = './result/GResNet/'
                result_obj.result_destination_file_name = 'DeepLoopyNetResNet_' + dataset_name + '_' + residual_type+'_depth_' + str(depth) + '_iter_' + str(iter)

                setting_obj = SettingCV('', '')

                evaluate_obj = EvaluateAcc('', '')
                #------------------------------------------------------

                #---- running section ---------------------------------
                setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
                acc_test = setting_obj.load_run_save_evaluate()
                print('Testing Acc: ', acc_test)
                acc_test_list.append(acc_test)
                print('*******************************')
                #------------------------------------------------------
            print(acc_test_list)
            print(np.mean(acc_test_list), np.std(acc_test_list))
            print('Finished')
