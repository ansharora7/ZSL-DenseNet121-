import os
import numpy as np
import time
import sys
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

print("GPU Available: "+str(torch.cuda.is_available()))

from ChexnetTrainer import ChexnetTrainer
from arguments import  parse_args


def main ():

    args = parse_args()
    # seed = 1002
    seed=1003
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:  
        os.mkdir(args.save_dir)  
    except OSError as error:
        print(error) 
    
    trainer = ChexnetTrainer(args)
    # checkpoint = torch.load('Densenet-finetuned.pth.tar')
    # trainer.model.load_state_dict(torch.load('Densnet-finetuned.pth'))
    trainer()

    checkpoint = torch.load(f'{args.save_dir}/min_loss_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    checkpoint2 = torch.load(f'{args.save_dir}/min_loss_vae-backprop.pth.tar')
    trainer.vae.load_state_dict(checkpoint2['state_dict'])
    print ('Testing the min loss model')
    test_ind_auroc = trainer.test()
    
    test_ind_auroc = np.array(test_ind_auroc)
    # test_ind_auroc_weighted = np.array(test_ind_auroc_weighted)
    # test_recallIndividual = np.array(test_recallIndividual)
    # test_precisionIndividual = np.array(test_precisionIndividual)
    # test_F1Individual = np.array(test_F1Individual)

# --beta-map 0.01 \
# --beta-con 0.01 \
# --neg-penalty 0.20 \

    trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    
    trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids],trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

    checkpoint = torch.load(f'{args.save_dir}/best_auroc_checkpoint.pth.tar')
    trainer.model.load_state_dict(checkpoint['state_dict'])
    checkpoint2 = torch.load(f'{args.save_dir}/best_auroc_vae-backprop.pth.tar')
    trainer.vae.load_state_dict(checkpoint2['state_dict'])
    print ('Testing the best AUROC model')
    test_ind_auroc = trainer.test()
    
    test_ind_auroc = np.array(test_ind_auroc)
    # test_ind_auroc_weighted = np.array(test_ind_auroc_weighted)
    # test_recallIndividual = np.array(test_recallIndividual)
    # test_precisionIndividual = np.array(test_precisionIndividual)
    # test_F1Individual = np.array(test_F1Individual)

    trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')

    # trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.seen_class_ids], test_ind_auroc_weighted[trainer.test_dl.dataset.seen_class_ids], 
    #                         test_recallIndividual[trainer.test_dl.dataset.seen_class_ids], test_precisionIndividual[trainer.test_dl.dataset.seen_class_ids], 
    #                         test_F1Individual[trainer.test_dl.dataset.seen_class_ids], trainer.test_dl.dataset.seen_class_ids, prefix='\ntest_seen')
    
    # trainer.print_auroc_new(test_ind_auroc[trainer.test_dl.dataset.unseen_class_ids], test_ind_auroc_weighted[trainer.test_dl.dataset.unseen_class_ids], 
    #                         test_recallIndividual[trainer.test_dl.dataset.unseen_class_ids], test_precisionIndividual[trainer.test_dl.dataset.unseen_class_ids], 
    #                         test_F1Individual[trainer.test_dl.dataset.unseen_class_ids], trainer.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen')




if __name__ == '__main__':
    main()





