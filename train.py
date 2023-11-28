
import os
import torch
import numpy as np
from src.nerf_dataset import load_nerf_data, create_dataloader
from src.models import NeRFModelRenderer, FastNeRF, NaiveNeRFModel
from src.trainer import NeRFTrainer

import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for NeRF models.")
    parser.add_argument(
        "--model",
        type=str,
        default="nerf",
        choices=["nerf", "fastnerf"], 
        help="Specify the model to train. Options are 'nerf', 'fastnerf', etc. [default: nerf]"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )

    ## TODO
    # load_checkpoint : to continue train from loaded model

    return parser.parse_args()

MODEL = {
    "nerf" : NaiveNeRFModel,
    "fastnerf": FastNeRF
}

def main():
    args = parse_args()

    Model = MODEL.get(args.model)
    if Model is None:
        print(f"Unsupported model name : {args.model}")
        exit(1)
        
    with open(args.config, 'r') as f:
        config = json.load(f)

        

    # Hyperparameters
    tn = config["tn"]
    tf = config["tf"]
    n_bins = config["n_bins"]
    n_epochs = config["n_epochs"]
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    img_w, img_h = config["image_size"]

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    obj_name = config["obj_name"]
    path = config["dataset_path"]
    data_path = os.path.join(os.path.join(path, config["dataset_type"]),f"{obj_name}")


    train_rays_o, train_rays_d, train_pixels, pose, (w, h, f) = load_nerf_data(data_path, "train", obj_name, (img_w, img_h))
    test_rays_o, test_rays_d, test_pixels, test_pose, _ = load_nerf_data(data_path, "test", obj_name, (img_w, img_h))

    data_loader = create_dataloader(train_rays_o, train_rays_d, train_pixels, batch_size=batch_size, shuffle=True)

    
    model = Model()
    loss_func = torch.nn.MSELoss()
    optimizer_builder = lambda model: torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir=config["checkpoint_dir"]
    checkpoint_file=config["checkpoint_file"]
    checkpoint_period=config["checkpoint_period"]

    ## train

    trainer = NeRFTrainer(model, NeRFModelRenderer, optimizer_builder, loss_func, checkpoint_dir=checkpoint_dir)
    trainer.train(data_loader, (img_w, img_h), tn, tf, n_bins, n_epochs, device=device, reset=True, cp_filename=checkpoint_file, cp_period=checkpoint_period, print_progress=True)

        


if __name__ == "__main__":
    main()
