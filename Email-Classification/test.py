import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from model import NeuralNet 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model data
FILE = "trained_data.pth"
data = torch.load(FILE)

# Extract model data
model_state = data["model_state"]
input_size = data["input_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

# Initialize the model
model = NeuralNet(input_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Define a function to predict the class of a given email text
def classify_email(email_text):
    # Convert email text to numerical data using the same TfidfVectorizer used during training
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True, vocabulary=all_words)
    email_text_numeric = vectorizer.fit_transform([email_text]).toarray()

    # Convert numerical data to PyTorch tensor
    input_tensor = torch.tensor(email_text_numeric, dtype=torch.float32)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output.data, 1)
        predicted_class = tags[predicted.item()]  # Map the predicted index to the corresponding class label
        return predicted_class

# Example usage:
email_text = "You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p p£3.99"
predicted_class = classify_email(email_text)
print("Predicted class:", predicted_class)
