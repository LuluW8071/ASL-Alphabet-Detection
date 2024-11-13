import comet_ml
import os 
import torch 
import dataset, engine, model
import utils
from torchvision import transforms
from comet_ml import Experiment

# Load API
from dotenv import load_dotenv
load_dotenv()

# Initialize Comet ML Experiment
comet_ml.login(api_key=os.getenv('COMET_API_KEY'),
                project_name="American Sign Langauge")

experiment = comet_ml.Experiment()

# Setup Hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_EPOCHS = 25
LEARNING_RATE = 4e-5

# Setup directories
source_dir = "asl"

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
model = model.EfficientNetB0(num_classes=36).to(device)
model = torch.compile(model)


# Setup loss_fn and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start the training through engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             classes=class_names,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             experiment=experiment)

experiment.end()

# Save the model through utils.py 
utils.save_model(model=model,
                 target_dir="models",
                 model_name=f"efficientnet_model.pth")