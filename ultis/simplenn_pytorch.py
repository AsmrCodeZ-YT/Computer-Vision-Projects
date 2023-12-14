import torch
import torch.nn as nn 
import torch.optim as optim 


class SampleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SampleNN, self).__init__()
        
        #layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        #activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
input_size, hidden_size, output_size = 2,4,2

model = SampleNN(input_size, hidden_size, output_size)

if torch.cuda.is_available():
    model = model.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Preprocessing Data
y_one_hot = torch.nn.functional.one_hot(y)

if torch.cuda.is_available():
   X = X.cuda()
   y_one_hot = y_one_hot.cuda()

# Training Neural Network
epochs = 500
for epoch in range(epochs):
   optimizer.zero_grad()
   outputs = model(X)


# Compute loss
   loss = criterion(outputs, y)
   # Backward pass
   loss.backward()
   # Update weights
   optimizer.step()
    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
       print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')