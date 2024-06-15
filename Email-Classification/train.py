import pandas as pd
import glob
import torch
import torch.nn as nn
import torch.nn.functional as f

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from model import NeuralNet
#from langdetect import detect

# Function to load data from multiple CSV files
def load_data(directory_path):
    file_paths = glob.glob(f"{directory_path}/*.csv")
    dfs = [pd.read_csv(file) for file in file_paths]
    data = pd.concat(dfs, ignore_index=True)
    return data

# Specify the directory containing your CSV files
data_directory = "data" 

# Load data from multiple files
df = load_data(data_directory)
dataset = df.where((pd.notnull(df)), '')

#pick a manual seed for randomization
torch.manual_seed(41)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#change spam to 0 and ham to 1
dataset.loc[dataset['Category'] == 'spam','Category',] = 0
dataset.loc[dataset['Category'] == 'ham','Category',] = 1

# Separate features and target
X = dataset['Message'] # email
Y = dataset['Category'].astype(int) # type

#Convert text data to numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)  
X = vectorizer.fit_transform(X).toarray()

#used to divide your dataset into training and testing subsets(0.2=>test 20% and training data 80%)
X_train, X_test, Y_train, Y_test =  train_test_split(X,Y,test_size = 0.2,random_state= 3 )

#convert x and y features to float tensors
#Convert the numerical data into PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

Y_train = torch.tensor(Y_train.values, dtype=torch.long)
Y_test = torch.tensor(Y_test.values, dtype=torch.long)

# Hyper-parameters 
learning_rate = 0.001
num_epochs = 10
X_train_dim = X_train.shape[1]

#create an instance of model
model = NeuralNet(X_train_dim).to(device)

#Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()#is used for multi-class classification problems. It combines 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train the model
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Clear the gradients
    outputs = model(X_train)  # Forward pass get predictions.
    loss = criterion(outputs, Y_train)  # calculates the loss between the model's predictions and the true labels.
    loss.backward()  # Backward pass to compute the gradients
    optimizer.step()  # Update the model parameters
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Print the loss for this epoch

#Evaluate the model
model.eval()#sets the model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total = Y_test.size(0)
    correct = (predicted == Y_test).sum().item()
    accuracy = correct / total
    print(f'Accuracy of the model on the test set: {accuracy * 100:.2f}%')

# Save the trained model
data = {
    "model_state": model.state_dict(),
    "input_size": X_train_dim,
    "output_size": 2,  # Assuming 2 classes: spam and ham
    "all_words": vectorizer.get_feature_names_out(),
    "tags": ['spam', 'ham']  # Assuming the classes are labeled as 'spam' and 'ham'
}

FILE = "trained_data.pth"
torch.save(data, FILE)

print(f'Training complete. Model saved to {FILE}')