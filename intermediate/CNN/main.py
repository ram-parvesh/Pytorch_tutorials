import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper_parameter
num_epochs = 5
num_classes = 10
batch_size =100
learning_rate = 0.001

# MNIST Datasets
train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

#DataLoader provides queue and threads
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

#convolutional Neural Network (two convolutional neural Network)
class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN,self).__init__()
        
        self.layer1 = nn.Sequential(                    #input size (1,28,28)
            nn.Conv2d(1, 16, kernel_size=5, padding=2), #in_channel=1,out_channel=16,filter size=5
            nn.BatchNorm2d(16),     # o/p size (16,28,28)
            nn.ReLU(),              #activation  o/p size  (16,28,28)
            nn.MaxPool2d(2))        #output size after pooling (16,14,14)
        
        
        self.layer2 = nn.Sequential(    #input shape (16,14,14)
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))          #output shape (32,7,7)
        
        self.fc = nn.Linear(32*7*7, num_classes)     # fully connected Layer,output 10 classes
    
    def forward(self,x):
        
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.view(out.size(0),-1)    #flatten the output of conv2d to feed into fully connected layers
        
        out = self.fc(out)
        return out 

    model = CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model 
total_step = len(train_loader)
print("total_step :",total_step)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate (train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass 
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward and optimize 
        optimizer.zero_grad() # clear gradient for this steps
        loss.backward()     # calculate gradient
        optimizer.step()    #apply gradients

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}],Step [{}/{}],loss:{:.4f}'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item()))  

# test the model 
model.eval()    # eval model (batchnorm uses moving mean./varience instead of mini-batch mean and varience)
with torch.no_grad():
    correct = 0
    total = 0
    for images ,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print("outputs data",outputs.data)
        _ ,predicted =torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')