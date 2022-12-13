import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper_parameter
num_epochs = 5
num_classes = 10
batch_size = 200
learning_rate = 0.01

# Mnist Datasets
train_datasets = dsets.MNIST(root ='./data',
                            train =True,
                            transform = transforms.ToTensor(),
                            download = True )
test_datasets = dsets.MNIST(root='./data',
                            train = False,
                            transform=transforms.ToTensor())

# Dataloaders provide queue and threads
train_loader = torch.utils.data.DataLoader(dataset = train_datasets,
                                            batch_size = batch_size,
                                            shuffle = True )

test_loader = torch.utils.data.DataLoader(dataset = test_datasets,
                                            batch_size = batch_size,
                                            shuffle = False)

# convolutional Neural Network
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN ,self).__init__()

        self.layer1 = nn.Sequential(
                                        nn.Conv2d(1,16,kernel_size = 5,padding = 2),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(16,32,kernel_size=5,padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2))
        self.fc = nn.Linear(32*7*7,num_classes)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.view(out.size(0),-1)

        out = self.fc(out)

        return out


model_SGD = CNN(num_classes).to(device)

model_Momentum = CNN(num_classes).to(device)

model_RMSprop = CNN(num_classes).to(device)

model_Adam = CNN(num_classes).to(device)

models = [model_SGD,model_Momentum,model_RMSprop,model_Adam]

# optimizers
opt_SGD = torch.optim.SGD(model_SGD.parameters(),lr=learning_rate)

opt_Momentum = torch.optim.SGD(model_Momentum.parameters(),lr=learning_rate,momentum=0.8)

opt_RMSprop = torch.optim.RMSprop(model_RMSprop.parameters(),lr=learning_rate,alpha=0.9)

opt_Adam = torch.optim.Adam(model_Adam.parameters(),lr=learning_rate,betas=(0.9,0.99))

optimizers = [opt_SGD,opt_Momentum,opt_RMSprop,opt_Adam]



# loss function 
criterion = nn.CrossEntropyLoss()


losses_his = [[],[],[],[]] 

total_step = len(train_loader)
print('total_step:',total_step)

for epoch in range(num_epochs):
    print("Epoch :",epoch)
    for idx,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        for model ,opt ,l_lis in zip(models,optimizers,losses_his):
            outputs = model(images)

            loss=criterion (outputs,labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

            l_lis.append(loss.item())



labels = ['SGD','Momentum','RMSprop','Adam']
for i ,l_lis in enumerate(losses_his):
    plt.plot(l_lis,label=labels[i])
plt.legend(loc= 'best')
plt.xlabel('Steps')
plt.ylabel('Loss')
# plt.ylim((0,0.2))
plt.show() 