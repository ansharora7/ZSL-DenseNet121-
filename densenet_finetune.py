from __future__ import print_function
from __future__ import division
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import time
import os
import copy
from dataset import NIHChestXray
from torch.utils.data import DataLoader
from arguments import  parse_args

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

print("GPU Available: "+str(torch.cuda.is_available()))

args = parse_args()
seed = 1002
torch.manual_seed(seed)
np.random.seed(seed)


class fine_tune(object):
    def __init__(self,args):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.args = args

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                
        train_transforms = []
        train_transforms.append(transforms.RandomResizedCrop(self.args.crop))
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.ToTensor())
        train_transforms.append(normalize)

        datasetTrain = NIHChestXray(self.args, self.args.train_file, transform=transforms.Compose(train_transforms))
        self.train_dl = DataLoader(dataset=datasetTrain, batch_size=self.args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
        
        
        # test_transforms = []
        # test_transforms.append(transforms.Resize(self.args.resize))
        # test_transforms.append(transforms.TenCrop(self.args.crop))
        # test_transforms.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # test_transforms.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        
        datasetVal =   NIHChestXray(self.args, self.args.val_file, transform=transforms.Compose(train_transforms))
        self.val_dl = DataLoader(dataset=datasetVal, batch_size=self.args.batch_size*10, shuffle=False, num_workers=4, pin_memory=True)


        feature_extract = False
        use_pretrained = True

        self.model_ft = models.densenet121(pretrained=use_pretrained)
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.model_ft, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features
            setattr(self.model_ft, classifier, nn.Identity(d_visual))
            break

        #self.model_ft.classifier = nn.Identity(d_visual)

        self.classifier = nn.Sequential(nn.Linear(d_visual, self.args.num_classes), nn.Sigmoid())

        # self.set_parameter_requires_grad(self.model_ft, feature_extract)

        self.model_ft = self.model_ft.to(self.device)
        self.classifier = self.classifier.to(self.device)

        
        # params_to_update = self.model_ft.parameters()
        
        # print("Params to learn:")
        # if feature_extract:
        #     params_to_update = []
        #     for name,param in self.model_ft.named_parameters():
        #         if param.requires_grad == True:
        #             params_to_update.append(param)
        #             print("\t",name)
        # else:
        #     for name,param in self.model_ft.named_parameters():
        #         if param.requires_grad == True:
        #             print("\t",name)

        # Observe that all parameters are being optimized
        self.optimizer_ft = optim.Adam (self.model_ft.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        # Setup the loss fxn
        self.criterion = nn.BCELoss(size_average=True)

        # Train and evaluate
        #self.train_model(self.criterion, self.optimizer_ft, num_epochs=100)


    
    
    def train_model(self,criterion, optimizer, num_epochs):
        file = open('results-new.txt','w')
        since = time.time()

        val_loss_history = []
        min_loss = float('inf')

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                print(phase)
                if phase == 'train':
                    self.model_ft.train()  # Set model to training mode
                    dataloader = self.train_dl
                else:
                    self.model_ft.eval()   # Set model to evaluate mode
                    dataloader = self.val_dl

                running_loss = 0.0
                # running_corrects = 0

                # Iterate over data.
                for _, (inputs, labels) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        outputs = self.model_ft(inputs)
                        preds = self.classifier(outputs)

                        if labels is not None:
                            loss = criterion(preds, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    #running_corrects += torch.sum(preds == labels.data)


                epoch_loss = running_loss / len(dataloader.dataset)
                #epoch_acc = running_corrects.double() / len(dataloader.dataset)

                print('{} Loss: {:.4f} Epoch: {}'.format(phase, epoch_loss,epoch))
                file.write('{} Loss: {:.4f} Epoch: {}'.format(phase, epoch_loss,epoch))
                file.write('\n')

                # deep copy the model
                # if phase == 'val' and epoch_acc > best_acc:
                #     best_acc = epoch_acc
                #     best_model_wts = copy.deepcopy(self.model_ft.state_dict())
                # if phase == 'val':
                #     val_acc_history.append(epoch_acc)

                if phase == 'val' and epoch_loss < min_loss:
                    min_loss = epoch_loss
                    self.save_checkpoint()
                    # torch.save(self.model_ft.state_dict(), 'Densnet-finetuned.pth')
                if phase == 'val':
                    val_loss_history.append(epoch_loss)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val loss: {:4f}'.format(min_loss))

        # load best model weights
        # self.model_ft.load_state_dict(best_model_wts)
        # return self.model_ft, val_loss_history

    
    def save_checkpoint(self):
        path = 'Densenet-finetuned.pth.tar'
        torch.save(
            { 
            'state_dict': self.model_ft.state_dict(),  
            'optimizer' : self.optimizer_ft.state_dict(),
            }, path)
        print("saving  checkpoint") 
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False



#Start Fine-tuning
k = fine_tune(args)
# from collections import OrderedDict
# state_dict = torch.load('Densnet-finetuned.pth')
# new_state_dict = OrderedDict()


# for key, value in state_dict.items():
#         new_key = 'vision_backbone.'+key
#         new_state_dict[new_key] = value

# torch.save(new_state_dict, 'Densnet-finetuned.pth')