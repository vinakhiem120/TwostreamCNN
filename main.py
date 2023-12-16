from runners.runner import runner
from tensorboardX import SummaryWriter
import yaml
import torch 
from utils.utils import dict_to_namespace
from torchvision.transforms import v2,InterpolationMode
import wandb

def main():
    logger = wandb
    transform = v2.Compose([
        v2.Resize(size=(226, 226), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5], std=[0.5])  # Gray normalization
    ])

    config_path = './configs/config.yml'

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = dict_to_namespace(config)  # Convert dictionary to Namespace object

    model = runner(config = config,logger = logger,transform= transform)
    model.train()


if __name__ == "__main__":
    main()