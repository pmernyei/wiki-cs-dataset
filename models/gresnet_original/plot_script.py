import matplotlib.pyplot as plt
from ResultSaving import ResultSaving

#---- biased GCN -----
if 0:
    residual_type = 'none'
    depth_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = '../../result/GResNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'DeepGCN_bias_depth_' + str(depth) + '_iter_' + str(0)
        depth_result_dict[depth] = result_obj.load()

    x = range(1000)

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        if residual_type == 'none':
            plt.plot(x, train_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, train_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
    plt.xlim(0, 1000)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        if residual_type == 'none':
            plt.plot(x, test_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, test_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)
    plt.xlim(0, 1000)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    print(best_score)


#--------------- GCN --------------

dataset_name = 'cora'

if 0:
    residual_type = 'naive'
    depth_list = [1, 2, 3, 4, 5, 6, 7]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GResNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'DeepGCNResNet_' + dataset_name + '_' + residual_type+'_depth_' + str(depth) + '_iter_' + str(0)
        depth_result_dict[depth] = result_obj.load()

    x = range(1000)

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        if residual_type == 'none':
            plt.plot(x, train_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, train_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
    plt.xlim(0, 1000)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(4, 3))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        if residual_type == 'none':
            plt.plot(x, test_acc, label='GCN(' + str(depth) + '-layer)')
        else:
            plt.plot(x, test_acc, label='GResNet(GCN,' + residual_type + ',' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)
    plt.xlim(0, 1000)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    print(best_score)

#--------------- GAT ---------------

#---- GAT naive residual ----
if 0:
    residual_type = 'graph_raw'
    epoch_number = 500
    depth_list = [1, 2, 3, 4, 5, 6, 7]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GResNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'DeepGATResNet_' + dataset_name + '_' +residual_type+'_depth_' + str(depth) + '_iter_' + str(0)
        depth_result_dict[depth] = result_obj.load()

    x = range(epoch_number)

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        plt.plot(x, train_acc, label='GResNet(GAT,' + residual_type + ',' + str(depth) + '-layer)')
    plt.xlim(0, epoch_number)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        plt.plot(x, test_acc, label='GResNet(GAT,' + residual_type + ',' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)
    plt.xlim(0, epoch_number)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    print(best_score)

#--------------- LoopyNet --------------

if 0:
    residual_type = 'graph_raw'
    depth_list = [1, 2, 3, 4, 5, 6, 7]
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = './result/GResNet/'
    best_score = {}

    depth_result_dict = {}
    for depth in depth_list:
        result_obj.result_destination_file_name = 'DeepLoopyNetResNet_' + dataset_name + '_' +residual_type+'_depth_' + str(depth) + '_iter_' + str(0)
        depth_result_dict[depth] = result_obj.load()

    x = range(1000)

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        print(depth_result_dict[depth].keys())
        train_acc = [depth_result_dict[depth]['learning_record'][i]['acc_train'] for i in x]
        if residual_type == 'none':
            plt.plot(x, train_acc, label='LoopyNet(' + str(depth) + '-layer)')
        else:
            plt.plot(x, train_acc, label='GResNet(LoopyNet,' + residual_type + ',' + str(depth) + '-layer)')
    plt.xlim(0, 1000)
    plt.ylabel("training accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(figsize=(5, 4))
    for depth in depth_list:
        test_acc = [depth_result_dict[depth]['learning_record'][i]['acc_test'] for i in x]
        if residual_type == 'none':
            plt.plot(x, test_acc, label='LoopyNet(' + str(depth) + '-layer)')
        else:
            plt.plot(x, test_acc, label='GResNet(LoopyNet,' + residual_type + ',' + str(depth) + '-layer)')
        best_score[depth] = max(test_acc)
    plt.xlim(0, 1000)
    plt.ylabel("testing accuracy %")
    plt.xlabel("epoch (iter. over training set)")
    plt.legend(loc="lower right")
    plt.show()

    print(best_score)
