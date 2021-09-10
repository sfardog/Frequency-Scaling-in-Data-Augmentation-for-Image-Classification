import os

import numpy as np

from .get_functions import get_save_path

def save_result(args, test_result, eecp=None) :
    if eecp :
        model_name = '{}(k_clusters:{}, length:{}, angle:{}, lamb:{}, preserve_range:{}, weight_factor : {}, combination:{})'.format(
            args.model_name, args.k_clusters, args.length, args.angle, args.lamb, args.preserve_range, args.weight_factor, args.combination)
    else :
        model_name = args.model_name

    model_dirs, save_model_path = get_save_path(args)

    save_path = os.path.join(model_dirs, '{} Results_{}.txt'.format(model_name, save_model_path))
    print("Your final result is saved from {}.".format(save_path))

    test_loss = test_result[0]
    test_top1_acc = test_result[1]

    print("###################### TEST REPORT ######################")
    print("FINAL TEST test loss : {} | test top1 error : {}".format(
        test_loss, 1 - test_top1_acc ))
    print("###################### TEST REPORT ######################")

    f = open(save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    f.write("test loss : {}\n".format(test_loss))
    f.write("test test top1 error : {}\n".format(np.round(1 - test_top1_acc, 6)))
    f.write("###################### TEST REPORT ######################")

    f.close()