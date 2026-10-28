import torch
import torch.nn as nn
from torchvision import datasets
#Changing from picture to torch tensors
from torchvision import transforms
#Visual
import matplotlib.pyplot as plt
#Files
from pathlib import Path
import numpy as np
#Rendering
from PIL import Image

############################# MISC #####################################
device = torch.device("mps")
epochs = 20
file = 'pixil-frame-7.png'
lr = .00065
root = Path(__file__).parent.resolve()
mnist = datasets.MNIST(root=root,download=True, train=True)
def prepare_png_for_mnist(path):
    # 1. Open the image and convert to Grayscale ('L' mode)
    img = Image.open(path).convert('L')
    
    # 2. Resize to 28x28 pixels
    img = img.resize((28, 28))
    
    # 3. Convert to a numpy array
    img_array = np.array(img)
    
    # 4. Invert colors if your source has a white background and black text
    # (MNIST requires a 0/black background and 255/white digit)
    img_array = 255 - img_array
    
    # 5. Normalize pixel values to a 0.0 - 1.0 range
    img_array = img_array / 255.0
    
    # 6. Reshape to add the batch and channel dimensions: (1, 28, 28, 1)
    img_tensor = img_array.reshape(28, 28)
    
    return img_tensor

############################# TRAINING #####################################

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.dataset = data
        #Rules for converting from image to tensor
        self.transform  = transforms.Compose([
                        #Converts AND normalizes the values
                        transforms.ToTensor(),
                        #Normalizes them even more around the means of x and y of MNIST
                        transforms.Normalize((0.1307,), (0.3081,))
                ])  
        
    #Dunder method, good for when you just call the class itself
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        #Split image and the label given to it
        image, label = sample[0], sample[1]
        #Returns the TRANSFORMED tensor from the image data and the tensor for the label
        return self.transform(image), torch.tensor(label)

    #Dunder method again, used when len() is used
    def __len__(self):
        return len(self.dataset)

#Make dataset (training = True)
data = Dataset(mnist)

#Load the dataset into batches of 64 pictures / batch
loader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=64,
        shuffle=True
)

#Child to the nn.Module main class. The main "AI" part.
class model(nn.Module):
    def __init__(self):
        super().__init__()
        #Make a tensor like this -> [[1,2],[3,4]]
        #Into this -> [1,2,3,4]
        self.flatten = nn.Flatten()
        #Make neurons with weights for this first layer
        #784 inputs (28 * 28 = 784) and we output 128 values
        self.hidden_layer = nn.Linear(
            in_features=784, out_features=128
        )
        #Second layer, takes the 128 inputs and makes it just 10 numbers
        #10 because each digit between 0-9 gets a "rank" and highest rank is the prediction
        self.hidden_layer2 = nn.Linear(
            in_features=128, out_features=10
        )
        #Make it all positive, convert negatives into 0 
        self.relu = nn.ReLU()
    
    #What is done every "step"
    def forward(self, x):
        #The given tensor is compressed into one big one
        flattened = self.flatten(x)
        #We do the first hidden linear layer
        #Outputs 128 values
        layer = self.hidden_layer(flattened)
        #We use ReLU to make it all positive
        relu_layer = self.relu(layer)
        #Run the second linear layer on the ReLU'd values from above
        layer2 = self.hidden_layer2(relu_layer)
        
        #Return the final values.
        #It returns a 10 value tensor, each digit containing a ranking for what number it predicts
        return layer2


learning_model = model().to(device)
#A type of loss calcuation, usually for complex data like pictures
loss = nn.CrossEntropyLoss()
#Optemizer to find the values that helped the model do what it did
optim = torch.optim.Adam(params=learning_model.parameters(), lr=lr)

#20 epochs
for epoch in range(epochs):
    averageLoss = 0
    #Split up data again
    for data, label in loader:
        data = data.to(device)
        label = label.to(device)
        
        #Train the model
        trained = learning_model(data)
        #Calcuate the loss
        loss_calcuated = loss(trained, label)
        #Reset the gradient
        optim.zero_grad()
        #Reverse engineer how it changed
        #If the loss went up, make the values in the OPPOSITE direction
        #If the loss went down, make the values in THAT direction
        loss_calcuated.backward()
        #Do the forward() step (linear layers, ReLU, etc...)
        optim.step()
    
        averageLoss += loss_calcuated.item()
        
    print(f'Loss: {averageLoss / len(loader):.4f}')
    
#Save to a file to load it
torch.save(learning_model.state_dict(), "mnist_model.pth")

############################# TESTING #####################################
learning_model = model().to(device)
learning_model.load_state_dict(
    torch.load("mnist_model.pth", map_location=device, weights_only=True)
)

#Turn off learning mode, evaluation mode
learning_model.eval()


#Make new dataset but for training (train = True)
mnist_new = datasets.MNIST(root=root,download=True, train=False)

#Load it up
data_new = Dataset(mnist_new)
loader_new = torch.utils.data.DataLoader(
        dataset=data_new,
        batch_size=64,
        shuffle=False
)


correct = 0
total = 0
with torch.no_grad():
    #Split up data, AGAIN
    for data, label in loader_new:
        data = data.to(device)
        label = label.to(device)
        
        #Run the trained model
        result = learning_model(data)
        
        #Find the predictions
        #dim = 1 makes it collapses the horozontals
        predictions = result.argmax(dim=1)
        
        #Check the total number of correct ones
        correct += (predictions == label).sum().item()
        total += len(label)
        
print(f"Test accuracy: {100 * correct / total:.2f}%")



#Misc function
ready_to_predict = prepare_png_for_mnist(file)
tensor = torch.tensor(ready_to_predict, dtype=torch.float32)
#Normalize (like we did before)
tensor = (tensor - 0.1307) / 0.3081
#Add [1,1,28,28] for matching up with other data
tensor = tensor.unsqueeze(0).unsqueeze(0)
tensor = tensor.to(device)



with torch.no_grad():
    result = learning_model(tensor)
    print(f'Predicted: {result.argmax(dim=1).item()} | Real: {file[len(file)-5]}')

#Display image
plt.imshow(tensor.cpu().squeeze(), cmap="gray")
plt.show()
