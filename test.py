import os
import numpy as np
import time
import torch
import sys
import statistics

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

print("GPU Available: "+str(torch.cuda.is_available()))

from ChexnetTrainer_Tester import ChexnetTrainer
from arguments import  parse_args

def main ():

    args = parse_args()
    
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)
    print ('Testing the trained model')
    

    # test_ind_auroc = trainer.test()
    # test_ind_auroc = np.array(test_ind_auroc)
    
    ###ADDED###
    test_ind_auroc, test_ind_auroc_weighted, test_recall, test_precision, test_F1 = trainer.test()
    test_ind_auroc = np.array(test_ind_auroc)
    test_ind_auroc_weighted = np.array(test_ind_auroc_weighted)
    test_recall = np.array(test_recall)
    test_precision = np.array(test_precision)
    test_F1 = np.array(test_F1)
    # print(test_ind_auroc.shape)
    # print(test_ind_auroc)
    # print(test_recall.shape)
    # print(test_recall)
    ###ADDED###


    # trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    # trainer.print_auroc(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

    ###ADDED###
    seen_aurocMean, seen_aurocMean_weighted, seen_recallMean, seen_precisionMean, seen_F1Mean = trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids],
                                                                                                                    test_ind_auroc_weighted[trainer.test_dl.dataset.seen_class_ids], 
                                                                                                                    test_recall[trainer.test_dl.dataset.seen_class_ids],
                                                                                                                    test_precision[trainer.test_dl.dataset.seen_class_ids],
                                                                                                                    test_F1[trainer.test_dl.dataset.seen_class_ids],
                                                                                                                    trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen',ver='test')
    
    unseen_aurocMean, unseen_aurocMean_weighted, unseen_recallMean, unseen_precisionMean, unseen_F1Mean = trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], 
                                                                                                            test_ind_auroc_weighted[trainer.test_dl.dataset.unseen_class_ids],
                                                                                                            test_recall[trainer.test_dl.dataset.unseen_class_ids],
                                                                                                            test_precision[trainer.test_dl.dataset.unseen_class_ids],
                                                                                                            test_F1[trainer.test_dl.dataset.unseen_class_ids],
                                                                                                            trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen',ver='test')

    trainer.write_results(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids],
                            test_ind_auroc_weighted[trainer.test_dl.dataset.seen_class_ids], 
                            test_recall[trainer.test_dl.dataset.seen_class_ids],
                            test_precision[trainer.test_dl.dataset.seen_class_ids],
                            test_F1[trainer.test_dl.dataset.seen_class_ids],
                            trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen',ver='test')


    trainer.write_results(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], 
                            test_ind_auroc_weighted[trainer.test_dl.dataset.unseen_class_ids],
                            test_recall[trainer.test_dl.dataset.unseen_class_ids],
                            test_precision[trainer.test_dl.dataset.unseen_class_ids],
                            test_F1[trainer.test_dl.dataset.unseen_class_ids],
                            trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen',ver='test')

    with open("output.txt", 'a') as results_file:
        results_file.write("\n\nHarmonic Means\n")
        results_file.write("AUROC: "+f'{harmonic_mean(seen_aurocMean, unseen_aurocMean):0.2f}\n')
        results_file.write("Weighted AUROC: "+f'{harmonic_mean(seen_aurocMean_weighted, unseen_aurocMean_weighted):0.2f}\n')
        results_file.write("Recall: "+f'{harmonic_mean(seen_recallMean, unseen_recallMean):0.2f}\n')
        results_file.write("Precision: "+f'{harmonic_mean(seen_precisionMean, unseen_precisionMean):0.2f}\n')
        results_file.write("F1: "+f'{harmonic_mean(seen_F1Mean, unseen_F1Mean):0.2f}\n')
    
    print("\n\nHarmonic Means")
    print("AUROC: "+f'{harmonic_mean(seen_aurocMean, unseen_aurocMean):0.2f}')
    print("Weighted AUROC: "+f'{harmonic_mean(seen_aurocMean_weighted, unseen_aurocMean_weighted):0.2f}')
    print("Recall: "+f'{harmonic_mean(seen_recallMean, unseen_recallMean):0.2f}')
    print("Precision: "+f'{harmonic_mean(seen_precisionMean, unseen_precisionMean):0.2f}')
    print("F1: "+f'{harmonic_mean(seen_F1Mean, unseen_F1Mean):0.2f}')
    ###ADDED###
            


def harmonic_mean(seen_metric, unseen_metric):
    return statistics.harmonic_mean([seen_metric, unseen_metric])
if __name__ == '__main__':
    main()





