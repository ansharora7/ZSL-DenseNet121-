import tensorflow as tf
import numpy as np
import os
import torchvision.transforms as transforms
from dataset import NIHChestXray
from tensorflow import keras
from tensorflow.keras import layers
from arguments import  parse_args
from torch.utils.data import DataLoader

args=parse_args()

# model = tf.keras.applications.densenet.DenseNet121(
#     include_top=False, weights='imagenet', input_tensor=None,
#     input_shape=None, pooling=None
# )

# DenseNet = keras.Sequential()
# DenseNet.add(model)
# DenseNet.add(layers.GlobalAveragePooling2D())


vision_backbone = getattr(torchvision.models,args.vision_backbone)(pretrained=args.pretrained)
# remove classification layer from visual encoder
classifiers = [ 'classifier', 'fc']
for classifier in classifiers:
    cls_layer = getattr(self.vision_backbone, classifier, None)
    if cls_layer is None:
        continue
    d_visual = cls_layer.in_features
    setattr(self.vision_backbone, classifier, nn.Identity(d_visual))
    break



normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
train_transforms = []
train_transforms.append(transforms.RandomResizedCrop(args.crop))
train_transforms.append(transforms.RandomHorizontalFlip())
train_transforms.append(transforms.ToTensor())
train_transforms.append(normalize)      

datasetTrain = NIHChestXray(args,args.train_file, transform=transforms.Compose(train_transforms))

train_dl = DataLoader(dataset=datasetTrain, batch_size=1, shuffle=True,  num_workers=4, pin_memory=True)

arr=[]

for batchID, (inputs, target) in enumerate (train_dl):
  op = DenseNet.predict(np.expand_dims(inputs,0))
  arr.append(op)

arr = np.array(arr)
np.save('DenseNet_new/DenseNet_features_new',arr)