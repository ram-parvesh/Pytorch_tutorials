import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 5
learning_rate = 0.003

#MNIST Datasets
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

# DataLoader  (provides queue and thread in a very simple way)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
                        
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False)

#Bidirectional recurrennt neural Network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self,input_size, hidden_size, num_layers,num_classes):
        super(BiRNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  #2 for bidirection

    def forward(self, x):
        #set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)  #2 for bidirection
        c0 = torch.zeros(self.num_layers*2 ,x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x,(h0,c0))  # tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the lat time step
        out = self.fc(out[:,-1,:])
        return out


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model

total_step = len(train_loader)

for epoch in range (num_epochs):
    for i ,(images ,labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # forward pass 
        outputs = model(images)
        loss = criterion(outputs,labels)

        # backward and optimize
        optimizer.zero_grad()    # clear gradient for this steps
        loss.backward()     # calculate gradient
        optimizer.step()     #apply gradient

        if (i+1) %100 ==0:
            print('Epoch :[{} /{}],Step [{}/{}],Loss: {:.4f}'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item()))

# Test the model 
with torch.no_grad():
    correct = 0
    total = 0
    for images ,labels in test_loader:
        images = images.reshape(-1,sequence_length,input_size).to(device)
        labels = labels.to(device)

        outputs=model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')