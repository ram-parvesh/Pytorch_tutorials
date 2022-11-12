import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Hyperparameters

input_size = 28*28  #784
num_class = 10
num_epochs = 5
batch_size = 100
leraning_rate = 0.001

#MNIST Datasets (images and labels)
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

#DataLoader (input pipeline)
train_loader =torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle = False)

# Logistic Model
model = nn.Linear(input_size, num_class)

#loss and Optimizer
# nn.CrossEntropyLoss() compute softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=leraning_rate)

# Train the model
total_step = len(train_loader)
print("Total_step: ",total_step)

for epoch in range(num_epochs):
    for i ,(images,labels) in enumerate(train_loader):
        # Reshape image to (batch_size,input_size)
        images = images.reshape(-1,input_size)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward and optimize 
        optimizer.zero_grad()     #clear gradient for this training step
        loss.backward()           #compute gradient
        optimizer.step()           #apply gradient

        if (i+1) %100 ==0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model 
# In the test phase we dont need to compute gradient  (for memory efficiency)

with torch.no_grad():
    correct = 0
    total = 0
    for images , labels in test_loader:
        images = images.reshape(-1,input_size)
        outputs = model(images)

        _,predicted = torch.max(outputs.data , 1)
        total += labels.size(0)
        correct += (predicted ==labels).sum()


        print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
    
    # save the model checkpoint
    torch.save(model.state_dict(),'model.ckpt')