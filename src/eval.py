import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from segnet import segnet
from custom_datasets import CustomDatasetFromImagesEval
from PIL import Image
from utils import convert_state_dict
import numpy as np
if __name__ == "__main__":

    transformations = transforms.Compose([transforms.ToTensor()])

    custom_mnist_from_images =  \
        CustomDatasetFromImagesEval('../data/iris_segmentation/lists/eval.csv')

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_images,
                                                    batch_size=1,
                                                    shuffle=False,num_workers=8)

    model = segnet()
    ckpt = torch.load("./best_model_6_800.pkl")
    state = convert_state_dict(ckpt['model_state'])
    model.load_state_dict(state)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()-1))
    model.cuda()
    with torch.no_grad():
        for i, (images, labels, name) in enumerate(mn_dataset_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            #images = Variable(images)
            #labels = Variable(labels)
            # Forward pass
            outputs = model(images)
            #print(outputs.size())
            img_array = outputs.cpu().numpy()
            #img_array = np.amax(img_array,1)
            #img_array = np.maximum(img_array[0,0,:,:],img_array[0,1,:,:])
            img_array = np.squeeze(np.argmax(img_array, axis = 1))*255
            #print(output.max())
            Img = Image.fromarray(img_array,'L')  
            print('../output/'+''.join(name))
            Img.save('../output/'+''.join(name),'PNG')

    print('Finished Evaluating')