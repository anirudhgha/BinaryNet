import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import time
from torch.utils.data import DataLoader
from collections import OrderedDict

# Transform class
class RangeTransform:
    def __call__(self, x):
        return 2 * x - 1

# Updated Binary activation function matching original implementation
class BinaryTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, stochastic=True):
        """
        Forward pass with optional stochastic binarization.
        Args:
            input (Tensor): pre-activation input.
            stochastic (bool): if True, sample binary output stochastically.
                               Otherwise, use deterministic rounding.
        Returns:
            Tensor: Binarized output in {-1, +1}.
        """
        # Compute hard sigmoid: clip((x+1)/2, 0, 1)
        p = torch.clamp((input + 1) / 2, 0, 1)
        if stochastic:
            # Stochastic binarization: sample each element from Bernoulli(p)
            y = torch.bernoulli(p)
        else:
            # Deterministic binarization: round to nearest 0 or 1
            y = torch.round(p)
        # Map {0, 1} to {-1, +1}
        y = 2 * y - 1
        ctx.save_for_backward(input)
        ctx.stochastic = stochastic  # save flag (though not used in backward)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Straight-through estimator: propagate gradient only for |input| < 1.
        grad_input = grad_output.clone() * ((input > -1).float() * (input < 1).float())
        # No gradient for the stochastic flag.
        return grad_input, None

# Convenience wrapper:
binary_tanh_unit = BinaryTanh.apply

# Calculate Glorot scaling
def glorot_scale(input_dim, output_dim):
    return float(1.0 / np.sqrt(1.5 / (input_dim + output_dim)))

# Updated Binary Convolution Layer with Glorot scaling
class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, binary=True, stochastic=False, H=1.0, W_LR_scale="Glorot"):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.binary = binary
        self.stochastic = stochastic
        
        # Calculate H using Glorot if specified
        if H == "Glorot":
            input_dim = in_channels * kernel_size * kernel_size
            output_dim = out_channels * kernel_size * kernel_size
            self.H = float(np.sqrt(1.5 / (input_dim + output_dim)))
        else:
            self.H = H
            
        # Calculate W_LR_scale using Glorot if specified
        if W_LR_scale == "Glorot":
            input_dim = in_channels * kernel_size * kernel_size
            output_dim = out_channels * kernel_size * kernel_size
            self.W_LR_scale = glorot_scale(input_dim, output_dim)
        else:
            self.W_LR_scale = W_LR_scale
        
        # Initialize weights uniformly within [-H, H]
        if binary:
            self.weight.data.uniform_(-self.H, self.H)
        
    def forward(self, x):
        if self.binary:
            # Save original weights for gradient updates
            org_weight = self.weight.data.clone()
            # Binarize weights
            bin_weight = self.binarize_weights(self.weight)
            # Replace weights temporarily for forward pass
            self.weight.data = bin_weight
            result = super().forward(x)
            # Restore weights
            self.weight.data = org_weight
            return result
        else:
            return super().forward(x)
            
    def binarize_weights(self, weights):
        # Convert to binary weights following original implementation
        wb = torch.clamp((weights / self.H + 1) / 2, 0, 1)
        if self.stochastic:
            wb = torch.bernoulli(wb)
        else:  # Deterministic rounding
            wb = torch.round(wb)
        # Map [0,1] back to [-H,H]
        wb = (wb * 2 - 1) * self.H
        return wb

# Updated Binary Linear Layer with Glorot scaling
class BinaryLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, binary=True, stochastic=False, H=1.0, W_LR_scale="Glorot"):
        super().__init__(in_features, out_features, bias)
        self.binary = binary
        self.stochastic = stochastic
        
        # Calculate H using Glorot if specified
        if H == "Glorot":
            self.H = float(np.sqrt(1.5 / (in_features + out_features)))
        else:
            self.H = H
            
        # Calculate W_LR_scale using Glorot if specified
        if W_LR_scale == "Glorot":
            self.W_LR_scale = glorot_scale(in_features, out_features)
        else:
            self.W_LR_scale = W_LR_scale
        
        # Initialize weights uniformly within [-H, H]
        if binary:
            self.weight.data.uniform_(-self.H, self.H)
        
    def forward(self, x):
        if self.binary:
            org_weight = self.weight.data.clone()
            bin_weight = self.binarize_weights(self.weight)
            self.weight.data = bin_weight
            result = super().forward(x)
            self.weight.data = org_weight
            return result
        else:
            return super().forward(x)
            
    def binarize_weights(self, weights):
        wb = torch.clamp((weights / self.H + 1) / 2, 0, 1)
        if self.stochastic:
            wb = torch.bernoulli(wb)
        else:
            wb = torch.round(wb)
        wb = (wb * 2 - 1) * self.H
        return wb

# BatchNorm with parameters matching the original code
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, alpha=0.1, epsilon=1e-4):
        super().__init__(num_features, eps=epsilon, momentum=(1.0 - alpha))

# Custom optimizer with LR scaling and weight clipping
class BinaryOptimizer:
    def __init__(self, model, base_lr, binary=True):
        self.model = model
        self.binary = binary
        self.base_lr = base_lr
        
        # Create parameter groups
        binary_params = []
        other_params = []
        
        for module in model.modules():
            if isinstance(module, (BinaryConv2d, BinaryLinear)) and binary:
                # Binary weights get scaled learning rates
                binary_params.append({
                    'params': [module.weight],
                    'lr': base_lr * module.W_LR_scale
                })
                if module.bias is not None:
                    other_params.append(module.bias)
            elif not isinstance(module, (BinaryConv2d, BinaryLinear)) and hasattr(module, 'weight'):
                if module.weight is not None:
                    other_params.append(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    other_params.append(module.bias)
        
        # Group parameters
        param_groups = [{'params': other_params}]
        param_groups.extend(binary_params)
        
        # Create the Adam optimizer
        self.optimizer = optim.Adam(param_groups, lr=base_lr)
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        # Standard optimizer step
        self.optimizer.step()
        
        # Then apply weight clipping
        if self.binary:
            self.clip_weights()
    
    def clip_weights(self):
        """Implementation of clipping_scaling from binary_net.py"""
        for module in self.model.modules():
            if isinstance(module, (BinaryConv2d, BinaryLinear)) and module.binary:
                # Clip weights to [-H, H]
                module.weight.data.clamp_(-module.H, module.H)
    
    def set_lr(self, lr):
        """Update learning rate with proper scaling"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # Non-binary params
                param_group['lr'] = lr
            else:  # Binary params with scaled LR
                # Extract module to get its W_LR_scale
                module_found = False
                for module in self.model.modules():
                    if isinstance(module, (BinaryConv2d, BinaryLinear)) and \
                       id(module.weight) == id(param_group['params'][0]):
                        param_group['lr'] = lr * module.W_LR_scale
                        module_found = True
                        break
                if not module_found:
                    param_group['lr'] = lr

# CNN model
class CIFAR10BinaryNet(nn.Module):
    def __init__(self, binary=True, stochastic=False, H=1.0, W_LR_scale="Glorot"):
        super().__init__()
        self.binary = binary
        self.stochastic = stochastic
        self.H = H
        self.W_LR_scale = W_LR_scale
        
        # 128C3-128C3-P2
        self.conv1 = BinaryConv2d(3, 128, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn1 = BatchNorm(128, alpha=0.1, epsilon=1e-4)
        self.conv2 = BinaryConv2d(128, 128, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn2 = BatchNorm(128, alpha=0.1, epsilon=1e-4)
        
        # 256C3-256C3-P2
        self.conv3 = BinaryConv2d(128, 256, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn3 = BatchNorm(256, alpha=0.1, epsilon=1e-4)
        self.conv4 = BinaryConv2d(256, 256, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn4 = BatchNorm(256, alpha=0.1, epsilon=1e-4)
        
        # 512C3-512C3-P2
        self.conv5 = BinaryConv2d(256, 512, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn5 = BatchNorm(512, alpha=0.1, epsilon=1e-4)
        self.conv6 = BinaryConv2d(512, 512, kernel_size=3, stride=1, padding=1, 
                                  binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn6 = BatchNorm(512, alpha=0.1, epsilon=1e-4)
        
        # 1024FP-1024FP-10FP
        self.fc1 = BinaryLinear(512 * 4 * 4, 1024, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn_fc1 = nn.BatchNorm1d(1024, eps=1e-4, momentum=0.9)
        self.fc2 = BinaryLinear(1024, 1024, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn_fc2 = nn.BatchNorm1d(1024, eps=1e-4, momentum=0.9)
        self.fc3 = BinaryLinear(1024, 10, binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale)
        self.bn_fc3 = nn.BatchNorm1d(10, eps=1e-4, momentum=0.9)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = binary_tanh_unit(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.bn2(x)
        x = binary_tanh_unit(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn3(x)
        x = binary_tanh_unit(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.bn4(x)
        x = binary_tanh_unit(x)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn5(x)
        x = binary_tanh_unit(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = self.bn6(x)
        x = binary_tanh_unit(x)
        
        # Fully connected layers
        x = x.view(-1, 512 * 4 * 4)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = binary_tanh_unit(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = binary_tanh_unit(x)
        x = self.fc3(x)
        x = self.bn_fc3(x)
        
        return x

def main():
    # Hyperparameters
    batch_size = 50
    alpha = 0.1
    epsilon = 1e-4
    binary = True
    stochastic = False
    H = 1.0
    W_LR_scale = "Glorot"
    
    num_epochs = 500
    LR_start = 0.001
    LR_fin = 0.0000003
    LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    train_set_size = 45000
    shuffle_parts = 1
    
    print("batch_size =", batch_size)
    print("alpha =", alpha)
    print("epsilon =", epsilon)
    print("activation = binary_tanh_unit")
    print("binary =", binary)
    print("stochastic =", stochastic)
    print("H =", H)
    print("W_LR_scale =", W_LR_scale)
    print("num_epochs =", num_epochs)
    print("LR_start =", LR_start)
    print("LR_fin =", LR_fin)
    print("LR_decay =", LR_decay)
    print("train_set_size =", train_set_size)
    print("shuffle_parts =", shuffle_parts)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load and preprocess CIFAR10 dataset
    print('Loading CIFAR-10 dataset...')
    
    # Transform data to range [-1, 1] as in the original code
    transform = transforms.Compose([
        transforms.ToTensor(),
        RangeTransform()  # Use class instead of lambda
    ])
    
    # Load training data
    full_train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Split into train and validation sets
    train_size = train_set_size
    val_size = 50000 - train_size
    train_set, val_set = torch.utils.data.random_split(full_train_set, [train_size, val_size])
    
    # Load test data
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model, transfer to GPU if available
    print('Building the CNN...')
    model = CIFAR10BinaryNet(binary=binary, stochastic=stochastic, H=H, W_LR_scale=W_LR_scale).to(device)
    
    # Use custom optimizer with LR scaling and weight clipping
    optimizer = BinaryOptimizer(model, LR_start, binary=binary)
    
    # Squared hinge loss
    def squared_hinge_loss(output, target):
        # Convert targets from class indices to one-hot and scale to [-1, 1]
        one_hot_target = F.one_hot(target, 10).float().to(device) * 2 - 1
        # Squared hinge loss
        return torch.mean(torch.clamp(1.0 - one_hot_target * output, min=0) ** 2)
    
    print('Training...')
    
    def train_epoch(epoch):
        model.train()
        train_loss = 0
        current_lr = LR_start * (LR_decay ** epoch)
        optimizer.set_lr(current_lr)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = squared_hinge_loss(output, target)
            loss.backward()
            optimizer.step()  # includes weight clipping
            train_loss += loss.item()
            
        return train_loss / len(train_loader)
    
    def evaluate(loader):
        model.eval()
        loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss += squared_hinge_loss(output, target).item()
                
                # Accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                
        return loss / len(loader), 100.0 * (1 - correct / total)  # Return loss and error rate
    
    best_val_err = 100
    best_epoch = 0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        train_loss = train_epoch(epoch)
        
        # Evaluation
        val_loss, val_err = evaluate(val_loader)
        test_loss, test_err = evaluate(test_loader)
        
        # Update best validation error
        if val_err < best_val_err:
            best_val_err = val_err
            best_epoch = epoch + 1
            # Save model
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print statistics
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} took {epoch_duration:.2f}s")
        print(f"  LR:                            {LR_start * (LR_decay ** epoch):.8f}")
        print(f"  training loss:                 {train_loss:.6f}")
        print(f"  validation loss:               {val_loss:.6f}")
        print(f"  validation error rate:         {val_err:.2f}%")
        print(f"  best epoch:                    {best_epoch}")
        print(f"  best validation error rate:    {best_val_err:.2f}%")
        print(f"  test loss:                     {test_loss:.6f}")
        print(f"  test error rate:               {test_err:.2f}%")

if __name__ == '__main__':
    main()