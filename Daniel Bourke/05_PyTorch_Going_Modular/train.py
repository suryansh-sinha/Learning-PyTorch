"""
Trains a PyTorch image classification model using device-agnostic code
"""

import os
from timeit import default_timer as timer

import torch
from torchvision.transforms import v2 as transforms

import data_setup, engine, model_builder, utils

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transforms
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.PILToTensor(),
    transforms.ToDtype(torch.float32, scale=True),
])

# Create dataloaders and get class names
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir, test_dir, data_transform, BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(3, HIDDEN_UNITS, len(class_names)).to(device)

# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Start training with help from engine.py
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=device)

# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

# Save the model.
utils.save_model(model, "models", "going_modular_tinyvgg.pth")
