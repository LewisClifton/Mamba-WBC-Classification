import os
from datetime import datetime
import torch
import torch.distributed as dist

def save(model_weights, out_dir, metrics, hyperparameters):

    # Get date/time of saving
    date = datetime.now().strftime("%Y_%m_%d_%p%I_%M")

    # Create output directory
    out_dir = os.path.join(out_dir, f'{date}/')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Model and log paths
    model_path = os.path.join(out_dir, f'ViT.pth')
    log_path = os.path.join(out_dir, 'log.txt')

    # Save trained model
    torch.save(model_weights, model_path)
    print(f'Model saved to {model_path}.')

    

    # For writing dictionary contents to a file
    def write_dict_to_file(file, dict_):
        for k, v in dict_.items():
            if isinstance(v, list) and len(v) == 0: continue # avoids error
            file.write(f'{k}: {v}\n')

    # Save training log
    with open(log_path , 'w') as file:
        
        file.write(f'Date/time of creation: {date}\n')

        file.write('\nTraining metrics:\n')
        write_dict_to_file(file, metrics)

        file.write('\nHyperparameters:\n')
        write_dict_to_file(file, hyperparameters)
    print(f'Saved log to {log_path}.')


def setup_dist(rank, world_size):
     # Set up process group and gpu model
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
