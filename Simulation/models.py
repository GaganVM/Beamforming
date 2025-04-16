import torch
import torch.nn as nn
import torch.nn.functional as F

class Model1(nn.Module):
    def __init__(self, input_size):
        super(Model1, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 128)  

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        x = self.linear7(x)
        return x

class Model2(nn.Module):
    def __init__(self, input_size):
        super(Model2, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        self.linear5 = nn.Linear(512, 256)
        self.linear6 = nn.Linear(256, 128)
        self.linear7 = nn.Linear(128, 128)  

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.tanh(self.linear5(x))
        x = F.sigmoid(self.linear6(x))
        x = self.linear7(x)
        return x

class Model3(nn.Module):
    def __init__(self, input_size):
        super(Model3, self).__init__()
        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 512)
        self.linear4 = nn.Linear(512, 512)
        self.linear5 = nn.Linear(512, 128)
  

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.sigmoid(self.linear5(x))
        return x

class LSTMModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size):
        super(LSTMModel,self).__init__()
        self.lstm=nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.linear1=nn.Linear(hidden_size,int(hidden_size/2))
        self.linear2=nn.Linear(int(hidden_size/2),output_size)
        self.linear3=nn.Linear(128,128)
    
    def forward(self,x):
        h0=torch.zeros(self.lstm.num_layers,x.size(0),self.lstm.hidden_size).to(x.device)
        c0=torch.zeros(self.lstm.num_layers,x.size(0),self.lstm.hidden_size).to(x.device)

        out,_=self.lstm(x,(h0,c0))
        out=out[:,-1,:]
        out=F.tanh(self.linear1(out))
        out=F.sigmoid(self.linear2(out))
        out=self.linear3(out)

        return out
    
class LSTMModelWithMultiheadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads):
        super(LSTMModelWithMultiheadAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)
        self.linear1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.linear2 = nn.Linear(int(hidden_size/2), output_size)
        self.linear3 = nn.Linear(128, 128)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        
        out, _ = self.multihead_attention(out.transpose(0, 1), out.transpose(0, 1), out.transpose(0, 1))
        out = out.transpose(0, 1)

        out = F.relu(self.linear1(out[:, -1, :])) #tanh
        out = F.sigmoid(self.linear2(out))
        out = self.linear3(out)

        return out