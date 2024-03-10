import os 
import torch 
import dataset, engine, model
import utils
from torchinfo import summary
from torchvision import transforms

# Setup Hyperparameters
BATCH_SIZE = 192
NUM_WORKERS = os.cpu_count()
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Setup directories
source_dir = "dataset/ASL_Alphabet_Dataset/asl_alphabet_train"

# Setup device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create transformation for image
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Add horizontal flipping
    transforms.TrivialAugmentWide(num_magnitude_bins=21),
    transforms.ToTensor()
])


# Create DataLoaders through data_setup.py
train_dataloader, test_dataloader, class_names = dataset.create_dataloaders(
    source_dir, data_transform, BATCH_SIZE, NUM_WORKERS
)

# Initalize EfficientNetB0 model
model = model.EfficientNetB0(num_classes=29).to(device)

summary(model=model,
        input_size=(BATCH_SIZE, 3, 128, 128),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])


# Setup loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the training through engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model through utils.py 
utils.save_model(model=model,
                 target_dir="models",
                 model_name="efficientnet_model.pth")