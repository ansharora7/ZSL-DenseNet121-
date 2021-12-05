import numpy as np
import time
from datetime import datetime, timedelta

from numpy.lib.function_base import average
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics.ranking import roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score
from zsl_models import ZSLNet
from dataset import NIHChestXray
from torch.nn.functional import kl_div, softmax, log_softmax
from numpy import dot
from numpy.linalg import norm
from plots import plot_array
# #-------------------------------------------------------------------------------- 

    
class ChexnetTrainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.textual_embeddings = np.load(args.textual_embeddings)

        self.model = ZSLNet(self.args, self.textual_embeddings, self.device).to(self.device)
        self.model.load_state_dict(torch.load('Densnet-finetuned.pth'),strict=False)
        self.vae = self.model.vae.model
        
        self.optimizer = optim.Adam (list(self.model.parameters()) + list(self.vae.parameters()), 
                                    lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        # self.optimizer = optim.Adam (list(self.model.parameters()) + list(self.vae.parameters()), 
        #                             lr=3e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        # self.optimizer = optim.Adam (self.vae.parameters(), 
                                    #  lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        
        # self.optimizer  = optim.SGD(list(self.model.parameters()) + list(self.vae.parameters()), lr=3e-3, momentum=0.9, weight_decay=0.0001)
        # self.optimizer  = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=0.0001)
        # self.scheduler = self.step_lr
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, mode='min')
        

        self.loss = torch.nn.BCELoss(size_average=True)
        self.auroc_min_loss = 0.0

        self.start_epoch = 1
        self.lossMIN = float('inf')
        self.max_auroc_mean = float('-inf')
        self.best_epoch = 1
        
        self.val_losses = []
        print(self.model)
        print(self.optimizer)
        print(self.scheduler)
        print(self.loss)
        print(f'\n\nloaded imagenet weights {self.args.pretrained}\n\n\n')
        self.resume_from()
        self.load_from()
        self.init_dataset()
        self.steps = [int(step) for step in self.args.steps.split(',')]
        self.time_start = time.time()
        self.time_end = time.time()
        self.should_test = False
        self.model.class_ids_loaded = self.train_dl.dataset.class_ids_loaded


    def __call__(self):
        self.train()
    
    def step_lr(self, epoch):
        step = self.steps[0]
        for index, s in enumerate(self.steps):
            if epoch < s:
                break
            else:
                step = s

        lr = self.args.lr * (0.1 ** (epoch // step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_from(self):
        if self.args.load_from is not None:
            checkpoint = torch.load(self.args.load_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f'ZSL loaded checkpoint from {self.args.load_from}')

        if self.args.vae_load_from is not None:
            checkpoint = torch.load(self.args.vae_load_from)
            self.vae.load_state_dict(checkpoint['state_dict'])
            print(f'VAE loaded checkpoint from {self.args.vae_load_from}')

    def resume_from(self):
        if self.args.resume_from is not None:
            checkpoint = torch.load(self.args.resume_from)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.lossMIN = checkpoint['lossMIN']
            self.max_auroc_mean = checkpoint['max_auroc_mean']
            print(f'resuming training from epoch {self.start_epoch}')

    def save_checkpoint(self, prefix='best'):
        path_zsl = f'{self.args.save_dir}/{prefix}_checkpoint.pth.tar'
        path_vae = f'{prefix}_vae-backprop.pth.tar'

        torch.save(
            {
            'epoch': self.epoch, 
            'state_dict': self.model.state_dict(), 
            'max_auroc_mean': self.max_auroc_mean, 
            'optimizer' : self.optimizer.state_dict(),
            'lossMIN' : self.lossMIN
            }, path_zsl)

        torch.save(
            { 
            'state_dict': self.vae.state_dict(),  
            'optimizer' : self.optimizer.state_dict(),
            }, path_vae)
            
        print(f"saving {prefix} checkpoint")

        
    def init_dataset(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        train_transforms = []
        train_transforms.append(transforms.RandomResizedCrop(self.args.crop))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)      

        datasetTrain = NIHChestXray(self.args, self.args.train_file, transform=transforms.Compose(train_transforms))

        self.train_dl = DataLoader(dataset=datasetTrain, batch_size=self.args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)


        test_transforms = []
        test_transforms.append(transforms.Resize(self.args.resize))
        test_transforms.append(transforms.TenCrop(self.args.crop))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        

        datasetVal =   NIHChestXray(self.args, self.args.val_file, transform=transforms.Compose(test_transforms))
        self.val_dl = DataLoader(dataset=datasetVal, batch_size=self.args.batch_size*10, shuffle=False, num_workers=4, pin_memory=True)
                
        datasetTest = NIHChestXray(self.args, self.args.test_file, transform=transforms.Compose(test_transforms), classes_to_load='all')
        self.test_dl = DataLoader(dataset=datasetTest, batch_size=self.args.batch_size*3, num_workers=8, shuffle=False, pin_memory=True)
        
        ###ADDED###
        self.train_imdb = datasetTrain.dict_num_images
        self.val_imdb = datasetVal.dict_num_images
        self.test_imdb = datasetTest.dict_num_images
        # print(self.train_dl.dataset.class_ids_loaded)
        # print(self.val_imdb)
        # print(self.test_imdb)
        ###ADDED###
        print(datasetTest.CLASSES)
        

    def train (self):
        print(self.args.epochs)
        for self.epoch in range (self.start_epoch, self.args.epochs):

            self.epochTrain()
            lossVal, val_ind_auroc, val_ind_auroc_weighted, recallIndividual, precisionIndividual, F1Individual = self.epochVal()
            val_ind_auroc = np.array(val_ind_auroc)
            val_ind_auroc_weighted = np.array(val_ind_auroc_weighted)
            recallIndividual = np.array(recallIndividual)
            precisionIndividual = np.array(precisionIndividual)
            F1Individual = np.array(F1Individual)


            aurocMean = val_ind_auroc.mean()
            self.save_checkpoint(prefix=f'last_epoch')  
            self.should_test = False

            if aurocMean > self.max_auroc_mean:
                self.max_auroc_mean = aurocMean
                self.save_checkpoint(prefix='best_auroc')
                self.best_epoch = self.epoch
                self.should_test = True
            if lossVal < self.lossMIN:
                self.lossMIN = lossVal
                self.auroc_min_loss = aurocMean
                self.save_checkpoint(prefix='min_loss')
                self.should_test = True

            self.print_auroc_new(val_ind_auroc, val_ind_auroc_weighted, recallIndividual, precisionIndividual, F1Individual,self.val_dl.dataset.class_ids_loaded, prefix='val',ver='val')
            if self.should_test is True:
                test_ind_auroc, test_ind_auroc_weighted, test_recallIndividual, test_precisionIndividual, test_F1Individual = self.test()
                test_ind_auroc = np.array(test_ind_auroc)
                test_ind_auroc_weighted = np.array(test_ind_auroc_weighted)
                test_recallIndividual = np.array(test_recallIndividual)
                test_precisionIndividual = np.array(test_precisionIndividual)
                test_F1Individual = np.array(test_F1Individual)
               
                self.write_results(val_ind_auroc, val_ind_auroc_weighted, recallIndividual, precisionIndividual, F1Individual, 
                                    self.val_dl.dataset.class_ids_loaded, prefix=f'\n\nepoch {self.epoch}\nval', mode='a', ver='val')

                
                self.write_results(test_ind_auroc[self.test_dl.dataset.seen_class_ids], test_ind_auroc_weighted[self.test_dl.dataset.seen_class_ids], 
                                    test_recallIndividual[self.test_dl.dataset.seen_class_ids], test_precisionIndividual[self.test_dl.dataset.seen_class_ids], 
                                    test_F1Individual[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen', mode='a',ver='test')
                
                
                self.write_results(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], test_ind_auroc_weighted[self.test_dl.dataset.unseen_class_ids], 
                                    test_recallIndividual[self.test_dl.dataset.unseen_class_ids], test_precisionIndividual[self.test_dl.dataset.unseen_class_ids], 
                                    test_F1Individual[self.test_dl.dataset.unseen_class_ids], self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen', mode='a',ver='test')

                self.print_auroc_new(test_ind_auroc[self.test_dl.dataset.seen_class_ids],test_ind_auroc_weighted[self.test_dl.dataset.seen_class_ids], 
                                    test_recallIndividual[self.test_dl.dataset.seen_class_ids], test_precisionIndividual[self.test_dl.dataset.seen_class_ids], 
                                    test_F1Individual[self.test_dl.dataset.seen_class_ids], self.test_dl.dataset.seen_class_ids, prefix='\ntest_seen',ver='test')
                
                
                self.print_auroc_new(test_ind_auroc[self.test_dl.dataset.unseen_class_ids], test_ind_auroc_weighted[self.test_dl.dataset.unseen_class_ids], 
                                    test_recallIndividual[self.test_dl.dataset.unseen_class_ids], test_precisionIndividual[self.test_dl.dataset.unseen_class_ids], 
                                    test_F1Individual[self.test_dl.dataset.unseen_class_ids],self.test_dl.dataset.unseen_class_ids, prefix='\ntest_unseen', ver='test')

            plot_array(self.val_losses, f'{self.args.save_dir}/val_loss')
            print(f'best epoch {self.best_epoch} best auroc {self.max_auroc_mean} loss {lossVal:.6f} auroc at min loss {self.auroc_min_loss:0.4f}')
            
            self.scheduler.step(lossVal)

                     
    #-------------------------------------------------------------------------------- 
    def get_eta(self, epoch, iter):
        self.time_end = time.time()
        delta = self.time_end - self.time_start
        delta = delta * (len(self.train_dl) * (self.args.epochs - epoch) - iter)
        sec = timedelta(seconds=int(delta))
        d = (datetime(1,1,1) + sec)
        eta = f"{d.day-1} Days {d.hour}:{d.minute}:{d.second}"
        self.time_start = time.time()

        return eta



    

    def epochTrain(self):
        
        self.model.train()
        self.vae.train()
        epoch_loss = 0
        for batchID, (inputs, target) in enumerate (self.train_dl):

            target = target.to(self.device)
            inputs = inputs.to(self.device)
            output, loss = self.model(inputs, target, self.epoch)


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            eta = self.get_eta(self.epoch, batchID)
            epoch_loss +=loss.item()
            if batchID % 10 == 9:
                print(f" epoch [{self.epoch:04d} / {self.args.epochs:04d}] eta: {eta:<20} [{batchID:04}/{len(self.train_dl)}] lr: \t{self.optimizer.param_groups[0]['lr']:0.4E} loss: \t{epoch_loss/batchID:0.5f}")


    #-------------------------------------------------------------------------------- 
        
    def epochVal (self):
        
        self.model.eval()
        self.vae.eval()
        
        lossVal = 0
        
        outGT = torch.FloatTensor().to(self.device)
        outPRED = torch.FloatTensor().to(self.device)
        for i, (inputs, target) in enumerate (tqdm(self.val_dl)):
            with torch.no_grad():
            
                target = target.to(self.device)
                inputs = inputs.to(self.device)
                varTarget = torch.autograd.Variable(target)    
                bs, n_crops, c, h, w = inputs.size()

                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))

                varOutput, losstensor = self.model(varInput, varTarget, n_crops=n_crops, bs=bs)


                

                outPRED = torch.cat((outPRED, varOutput), 0)
                outGT = torch.cat((outGT, target), 0)

                lossVal+=losstensor.item()
                del varOutput, varTarget, varInput, target, inputs
        lossVal = lossVal / len(self.val_dl)
        
        aurocIndividual, aurocIndividual_weighted = self.computeAUROC(outGT, outPRED, self.val_dl.dataset.class_ids_loaded)
        recallIndividual, precisionIndividual, F1Individual = self.computeMetrics(outGT, outPRED, 
                                                                        self.val_dl.dataset.class_ids_loaded) 
        
        self.val_losses.append(lossVal)

        return lossVal, aurocIndividual, aurocIndividual_weighted, recallIndividual, precisionIndividual, F1Individual
    
    
   
    def test(self):
        cudnn.benchmark = True
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        self.model.eval()
        self.vae.eval()
	
        
        for i, (inputs, target) in enumerate(tqdm(self.test_dl)):
            with torch.no_grad():
                target = target.to(self.device)
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = inputs.size()
                
                varInput = torch.autograd.Variable(inputs.view(-1, c, h, w).to(self.device))
                
                out, _ = self.model(varInput, n_crops=n_crops, bs=bs)
                # print(out)
                # print("OUT!!!!!!!!!!!!!!!!!!!!")
                
                outPRED = torch.cat((outPRED, out.data), 0)
                


        aurocIndividual, aurocIndividual_weighted = self.computeAUROC(outGT, outPRED, self.test_dl.dataset.class_ids_loaded)
        recallIndividual, precisionIndividual, F1Individual = self.computeMetrics(outGT, outPRED, 
                                                                        self.test_dl.dataset.class_ids_loaded) 
        
        aurocMean = np.array(aurocIndividual).mean()
        
        return aurocIndividual,aurocIndividual_weighted, recallIndividual, precisionIndividual, F1Individual
    
    
    def computeAUROC (self, dataGT, dataPRED, class_ids):
        outAUROC = []
        outAUROC_weighted = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()

        for i in class_ids:
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            # print("HI HI HI")
            # print(datanpGT[:, i])
            # print(np.round(datanpPRED[:,i]))
            # print(datanpPRED[:,i])
            # print('--------------')
            outAUROC_weighted.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i],average='weighted'))
        return outAUROC, outAUROC_weighted


    def computeMetrics (self, dataGT, dataPRED, class_ids):
        outRecall = []
        outPrecision = []
        outF1 = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        print(datanpPRED[0,:])
        print("YAYAYAYAYA")
        for i in class_ids:
            x = np.round(datanpPRED[:,i])
            x[x==0.]=0.
            # print(datanpGT[:,i].size)
            # print(datanpPRED[:,i].size)
            # print(datanpPRED[])
            # print(x)
            outRecall.append(recall_score(datanpGT[:, i], x))
            outPrecision.append(precision_score(datanpGT[:, i],x))
            outF1.append(f1_score(datanpGT[:, i], x))
        
        return outRecall, outPrecision, outF1


    def write_results(self, aurocIndividual, aurocIndividual_weighted, recallIndividual, precisionIndividual, F1Individual, class_ids, prefix='val', mode='a',ver='val'):

        with open("output.txt", mode) as results_file:
            aurocMean = aurocIndividual.mean()
            aurocMean_weighted = self.weighted_mean(aurocIndividual_weighted,class_ids,ver)
            recallMean = self.weighted_mean(recallIndividual,class_ids,ver)
            precisionMean = self.weighted_mean(precisionIndividual,class_ids,ver)
            F1Mean = self.weighted_mean(F1Individual,class_ids,ver)
                

            results_file.write(f'{prefix} AUROC mean {aurocMean:0.2f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.2f}\n')

            results_file.write(f'{prefix} Weighted AUROC mean {aurocMean_weighted:0.2f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual_weighted[i]:0.2f}\n')

            results_file.write(f'{prefix} Recall mean {recallMean:0.2f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {recallIndividual[i]:0.2f}\n')

            results_file.write(f'{prefix} Precision mean {precisionMean:0.2f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {precisionIndividual[i]:0.2f}\n')

            results_file.write(f'{prefix} F1 mean {F1Mean:0.2f}\n')
            for i, class_id in enumerate(class_ids):  
                results_file.write(f'{self.val_dl.dataset.CLASSES[class_id]} {F1Individual[i]:0.2f}\n')



    def weighted_mean(self, ind_rate, class_ids,ver):
        if(ver=='val'):
            imdb=self.val_imdb
        else:
            imdb=self.test_imdb
        ans=0
        total_wt=0
        for i, class_id in enumerate(class_ids):
            total_wt+=imdb.get(class_id)
            ans+=ind_rate[i]*imdb.get(class_id)
        return ans/total_wt


    
    def print_auroc(self, aurocIndividual, recallIndividual, precisionIndividual, F1Individual, class_ids, prefix='val'):
        aurocMean = aurocIndividual.mean()
        # aurocMean_weighted = self.weighted_mean(aurocIndividual_weighted,class_ids_auroc)
        recallMean = recallIndividual.mean()
        precisionMean = precisionIndividual.mean()
        F1Mean = F1Individual.mean()
        


        print (f'{prefix} AUROC mean {aurocMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print (f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.2f}')

        # print (f'{prefix} Weighted AUROC mean {aurocMean_weighted:0.4f}')
        # for i, class_id in enumerate(class_ids):  
        #     print (f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual_weighted[i]:0.4f}')

        print(f'{prefix} Recall mean {recallMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {recallIndividual[i]:0.2f}')

        print(f'{prefix} Precision mean {precisionMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {precisionIndividual[i]:0.2f}')

        print(f'{prefix} F1 mean {F1Mean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {F1Individual[i]:0.2f}')

        return aurocMean, recallMean, precisionMean, F1Mean






    def print_auroc_new(self, aurocIndividual, aurocIndividual_weighted, recallIndividual, precisionIndividual, F1Individual, class_ids, prefix='val',ver='val'):
        # print(aurocIndividual)
        # print(recallIndividual)
        # print(aurocIndividual_weighted)
        aurocMean = aurocIndividual.mean()
        aurocMean_weighted = self.weighted_mean(aurocIndividual_weighted,class_ids,ver)
        recallMean = self.weighted_mean(recallIndividual,class_ids,ver)
        precisionMean = self.weighted_mean(precisionIndividual,class_ids,ver)
        F1Mean = self.weighted_mean(F1Individual,class_ids,ver)
        


        print (f'{prefix} AUROC mean {aurocMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print (f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual[i]:0.2f}')
        # print('\n')

        print (f'{prefix} Weighted AUROC mean {aurocMean_weighted:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print (f'{self.val_dl.dataset.CLASSES[class_id]} {aurocIndividual_weighted[i]:0.2f}')
        # print('\n')

        print(f'{prefix} Recall mean {recallMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {recallIndividual[i]:0.2f}')
        # print('\n')

        print(f'{prefix} Precision mean {precisionMean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {precisionIndividual[i]:0.2f}')
        # print('\n')

        print(f'{prefix} F1 mean {F1Mean:0.2f}')
        for i, class_id in enumerate(class_ids):  
            print(f'{self.val_dl.dataset.CLASSES[class_id]} {F1Individual[i]:0.2f}')
        # print('\n')

        return aurocMean, aurocMean_weighted, recallMean, precisionMean, F1Mean
        


