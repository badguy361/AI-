import torch

class Net(torch.nn.Module):  
    def __init__(self, n_feature, n_output):  
        super(Net, self).__init__()  
        self.input = torch.nn.Linear(n_feature, 512)
        self.layer1 = torch.nn.Linear(512, 1024)
        self.layer2 = torch.nn.Linear(1024, 512)
        self.out = torch.nn.Linear(512, n_output)
      
    def forward(self, x_layer):  
        x_layer = torch.relu(self.input(x_layer))
        x_layer = torch.relu(self.layer1(x_layer)) 
        x_layer = torch.relu(self.layer2(x_layer)) 
        x_layer = self.out(x_layer) 
        # x_layer = torch.nn.functional.softmax(x_layer)  
        return x_layer