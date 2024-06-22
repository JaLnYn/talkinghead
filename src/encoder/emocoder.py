import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image

class Emocoder(nn.Module):
    def __init__(self):
        super(Emocoder, self).__init__()
        self.model = None
        self.model = models.resnet18(weights='IMAGENET1K_V1')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Unfreezing the last three layers
        for name, child in self.model.named_children():
            if name in ['layer3', 'layer4', 'fc']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
        
        # Adding custom layers
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512)
        )
        
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model.to(self.device)
        
        # Setting class weights for the loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def load_data(self, data_path, batch_size=32):
        dataset = datasets.ImageFolder(data_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader

    def train(self, data_path, num_epochs=5, learning_rate=0.001):

        self.model.train()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, momentum=0.9)
        train_loader = self.load_data(data_path)

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    def save_model(self, file_path='./models/portrait/emodel.pth'):
        torch.save(self.model.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path='./models/portrait/emodel.pth'):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        print(f'Model loaded from {file_path}')

    def predict(self, img_tensor):
        output = self.model(img_tensor)
        print("Output shape:", output.shape) 

def get_trainable_emonet():
    return Emocoder()


# Usage example
if __name__ == "__main__":
    emocoder = Emocoder(None)
    emocoder.train(num_epochs=3)
    emocoder.save_model()
    emocoder.load_model('resnet18_finetuned.pth')
    emocoder.predict('path_to_your_image.jpg')
