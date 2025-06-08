import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.get import GET_Classifier_models

# Settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_CLASSES = 10
INPUT_SIZE = 32
MODEL_NAME = 'GET-Classifier-T'  # or S/B
LR = 1e-3

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
data_iter = iter(loader)
images, labels = next(data_iter)
images, labels = images.to(device), labels.to(device)

# Model
class DummyArgs:
    mem = False
args = DummyArgs()
model = GET_Classifier_models[MODEL_NAME](args=args, input_size=INPUT_SIZE, num_classes=NUM_CLASSES).to(device)
model.train()
# Ensure all parameters and buffers are on the correct device
model.cls_token = nn.Parameter(model.cls_token.data.to(device))
model.pos_embed = nn.Parameter(model.pos_embed.data.to(device))

# --- Disable DEQ: Replace model's forward with a standard transformer stack ---
def forward_no_deq(self, x):
    # Patch embedding + positional encoding
    x = self.x_embedder(x)
    B = x.shape[0]
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    # Just run through the blocks sequentially
    for block in self.deq_blocks:
        x = block(x, None, None)
    x = self.norm_final(x)
    cls_token_final = x[:, 0]
    logits = self.head(cls_token_final)
    return logits

import types
model.forward = types.MethodType(forward_no_deq, model)
model = model.to(device)  # Ensure all submodules are on the correct device after patching

# Debug: print device of each block's parameters
for i, block in enumerate(model.deq_blocks):
    for name, param in block.named_parameters():
        print(f"Block {i} param {name} device: {param.device}")

# Optimizer and loss
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    output = model(images)
    if isinstance(output, list):
        loss = criterion(output[-1], labels)
        pred = output[-1].argmax(dim=1)
    else:
        loss = criterion(output, labels)
        pred = output.argmax(dim=1)
    acc = (pred == labels).float().mean().item() * 100
    loss.backward()
    optimizer.step()
    if step % 50 == 0 or acc == 100.0:
        print(f"Step {step}: Loss={loss.item():.4f}, Acc={acc:.2f}%")
    if acc == 100.0:
        print("Model has overfit the batch!")
        break 