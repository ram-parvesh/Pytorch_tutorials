# Dependencies 
# torch
# matplotlib
# trochvision

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt



# Hyper- Paramter
num_epochs = 5
num_classes = 10
batch_size = 28
time_step = 28   # rnn time step /image height
input_size = 28   # rnn input size / image width
learning_rate = 0.001   #learning rate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_datasets = dsets.MNIST(root='./data',
                            train =True,
                            transform=transforms.ToTensor(),
                            download=True)

test_datasets = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_datasets, 
                                            batch_size=batch_size,
                                            shuffle=True)


test_loader = torch.utils.data.DataLoader(dataset=test_datasets, 
                                            batch_size=batch_size,
                                            shuffle=True)


##### model

class RNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes =num_classes

                            #if use nn.RNN() , it hardly learns
        self.rnn = nn.LSTM(     
            input_size = input_size,
            hidden_size = 64,   #rnn hidden unit
            num_layers = 1,    #num of rnn layers
            batch_first = True,  #input and output will have batch size as ls dimension eg (batch_size ,time_step,input_size)
        )                                       

        self.out =nn.Linear(64 ,num_classes)
    
    def forward(self,x):
        # x shape (batch_size, time_step ,input_size)
        # r_out shape (batch_size, time_step, output_size)
        # h_n shape (n_layers , batch_size , hidden_size)
        # h_c shape (n_layers, batch_size, hidden_size)
        r_out , (h_n, h_c) = self.rnn(x, None) #none represents zero initial hidden state
 

        out = self.out(r_out[:,-1,:])

        return out

model = RNN(num_classes=num_classes).to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


# training
for epoch in range(num_epochs):
    for idx, (image ,label) in enumerate(train_loader):
        
        label =label.to(device)

        image = image.view(-1,28,28)  #reshape to (batch_size, time_step, input_size)
        image = image.to(device)

        output = model(image)

        loss = criterion(output, label) #calculate loss

        optimizer.zero_grad()   #clear grads
        loss.backward()         #calculate gradients
        optimizer.step()         # apply gradients

       
print("training completed")