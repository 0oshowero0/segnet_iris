import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from segnet_model import SegNet
from segnet import segnet
from custom_datasets import CustomDatasetFromImages
from loss import cross_entropy2d
if __name__ == "__main__":

    transformations = transforms.Compose([transforms.ToTensor()])

    custom_mnist_from_images =  \
        CustomDatasetFromImages('../data/iris_segmentation/lists/train.csv')

    mn_dataset_loader = torch.utils.data.DataLoader(dataset=custom_mnist_from_images,
                                                    batch_size=2,
                                                    shuffle=False,num_workers=8)

    #model = SegNet(num_classes=2)
    model = segnet()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()-1))
    model.cuda()
    criterion = cross_entropy2d
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
    loss_min = 100 
    for epoch in range(30):
        running_loss = 0.0
        for i, (images, labels) in enumerate(mn_dataset_loader):
            images = Variable(images.cuda())
            #images = Variable(images)
            #labels = Variable(labels.view(2,1024,1024))
            #labels = Variable(labels.view(2,1024,1024).cuda())
            labels = Variable(labels.cuda())
            # Clear gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(images)
            # Calculate loss
            #loss = F.binary_cross_entropy(F.sigmoid(input), labels)
            #loss = criterion(outputs, labels)
            loss = criterion(input=outputs, target=labels)
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
            running_loss += loss.item()
            print("%d" % i)
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
            if i % 100 == 99:    
                current_loss = loss.item()
                if current_loss < loss_min:
                    loss_min = current_loss
                    state = {'epoch': epoch+1,
                        'model_state': model.state_dict(),
                        'optimizer_state' : optimizer.state_dict(),}
                    torch.save(state, "./best_model_"+str(epoch+1)+"_"+str(i+1)+".pkl")

    print('Finished Training')

    