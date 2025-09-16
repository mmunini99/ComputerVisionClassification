import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from itertools import combinations
from sklearn.svm import SVC



class ShallowCNN(nn.Module):
    """Shallow CNN according to Table 1 specifications"""
    def __init__(self, num_classes=15):
        super(ShallowCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Fully connected layers
        # After 3 pooling operations: 64/8 = 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 15)
        
        # Initialize weights and biases
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Initialize weights from Gaussian distribution (mean=0, std=0.01)
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                # Initialize biases to 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)  # 64x64 -> 32x32
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Third conv block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Flatten for fully connected layer
        x = x.view(x.size(0), -1)
        
        # Fully connected layer 
        x = self.fc1(x)
        
        return x
    


class ImprovedCNN(nn.Module):
    """Improved CNN with batch normalization and dropout"""
    def __init__(self, num_classes=15):
        super(ImprovedCNN, self).__init__()
        
        # Convolutional layers with varying filter sizes
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    



class AlexNetTransfer(nn.Module):
    def __init__(self, num_classes, freeze_features=True):
        super(AlexNetTransfer, self).__init__()
        
        # Load pre-trained AlexNet
        self.alexnet = models.alexnet(pretrained=True)
        
        # Freeze feature extraction layers
        if freeze_features:
            for param in self.alexnet.features.parameters():
                param.requires_grad = False
            
            # Also freeze the first two fully connected layers
            for param in self.alexnet.classifier[:-1].parameters():
                param.requires_grad = False
        
        # Replace the last fully connected layer
        # AlexNet classifier has: Dropout, Linear(4096, 4096), ReLU, Dropout, Linear(4096, 4096), ReLU, Linear(4096, 1000)
        in_features = self.alexnet.classifier[-1].in_features
        self.alexnet.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.alexnet(x)


class AlexNetFeatureExtractor(nn.Module):
    def __init__(self, layer_name):
        """
        AlexNet feature extractor.
        layer_name options:
        - 'fc1': First fully connected layer (4096 features)
        - 'fc2': Second fully connected layer (4096 features)
        - 'features': Convolutional features (9216 features after adaptive pooling)
        """
        super(AlexNetFeatureExtractor, self).__init__()
        
        # Load pre-trained AlexNet
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.eval()  # Set to evaluation mode
        
        # Freeze all parameters
        for param in self.alexnet.parameters():
            param.requires_grad = False
        
        self.layer_name = layer_name
        self.features = None
        
        # Register forward hook to extract intermediate features
        if layer_name == 'fc1':
            # First fully connected layer (after dropout)
            self.alexnet.classifier[2].register_forward_hook(self._get_features)
        elif layer_name == 'fc2':
            # Second fully connected layer (after dropout)
            self.alexnet.classifier[5].register_forward_hook(self._get_features)
        elif layer_name == 'features':
            # Convolutional features
            self.alexnet.features.register_forward_hook(self._get_features)
    
    def _get_features(self, module, input, output):
        """Hook function to extract features"""
        if self.layer_name == 'features':
            # Apply adaptive pooling to get fixed size
            pooled = nn.AdaptiveAvgPool2d((6, 6))(output)
            self.features = pooled.view(pooled.size(0), -1).detach().cpu().numpy()
        else:
            self.features = output.detach().cpu().numpy()
    
    def forward(self, x):
        # Forward pass through AlexNet
        _ = self.alexnet(x)
        return self.features
    






class DAG_SVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Directed Acyclic Graph SVM for multiclass classification
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.binary_classifiers = {}
        self.classes_ = None
        self.n_classes = 0
        
    def fit(self, X, y):
        """Train binary classifiers for all class pairs"""
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        print(f"Training {self.n_classes * (self.n_classes - 1) // 2} binary SVM classifiers...")
        
        # Train binary classifier for each pair of classes
        for i, (class1, class2) in enumerate(combinations(self.classes_, 2)):
            print(f"Training classifier {i+1}/{self.n_classes * (self.n_classes - 1) // 2}: Class {class1} vs Class {class2}")
            
            # Create binary dataset
            mask = (y == class1) | (y == class2)
            X_binary = X[mask]
            y_binary = y[mask]
            
            # Convert to binary labels (0 and 1)
            y_binary = (y_binary == class2).astype(int)
            
            # Train binary SVM
            svm = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, probability=False)
            svm.fit(X_binary, y_binary)
            
            self.binary_classifiers[(class1, class2)] = svm
    
    def predict(self, X):
        """Predict using DAG structure"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            predictions[i] = self._predict_single(X[i:i+1])
        
        return predictions
    
    def _predict_single(self, x):
        """Predict single sample using DAG traversal"""
        remaining_classes = set(self.classes_)
        
        # DAG traversal: eliminate one class at each step
        while len(remaining_classes) > 1:
            # Pick the first two classes from remaining classes
            classes_list = sorted(list(remaining_classes))
            class1, class2 = classes_list[0], classes_list[1]
            
            # Get the corresponding binary classifier
            if (class1, class2) in self.binary_classifiers:
                classifier = self.binary_classifiers[(class1, class2)]
                prediction = classifier.predict(x)[0]
                
                # Remove the losing class
                if prediction == 0:  # class1 wins
                    remaining_classes.remove(class2)
                else:  # class2 wins
                    remaining_classes.remove(class1)
            else:
                # This shouldn't happen if we trained all pairs
                remaining_classes.remove(class2)
        
        return list(remaining_classes)[0]

