import os

import torch
import torch.distributed as dist


def save_log(out_dir, date, metrics, model_config, dataset_config):
    """
    Save training log

    Args:
        out_dir(string): Path to directory to save the log file to
        date(string): Date/time of training
        metrics(dict or list[dict]): Model or list of metrics dictionarys to be saved
    """
    
    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:
        file.write(f'Date/time of creation: {date}\n')

        if isinstance(metrics, list):
            # Save metrics for each fold
            for idx, fold_metrics in enumerate(metrics):
                file.write(f'\nFold {idx} training metrics:\n')
                write_dict_to_file(file, fold_metrics)
        else:
            # Save metrics for the model
            file.write(f'\nTraining metrics:\n')
            write_dict_to_file(file, metrics)

        file.write(f'\nModel configuration:\n')
        write_dict_to_file(file, model_config)

        file.write(f'\nDataset configuration:\n')
        write_dict_to_file(file, dataset_config)

    print(f'\nSaved log to {log_path}')


def write_dict_to_file(file, dict_):
    """
    Write a dictionary to a given file

    Args:
        file(TextIOWrapper[_WrappedBuffer]): File to be written to
        dict_(dict): Dictionary to be written to the file
    """
    
    for k, v in dict_.items():
        if isinstance(v, list) and len(v) in [0, 'cuda:0']: continue # avoids error
        file.write(f'{k}: {v}\n')


def setup_dist(rank, world_size):
    """
    Standard process group set up for Pytorch Distributed Data Parallel

    Args:
        rank(int): Id of the GPU used to call this function
        world_size(int): Total number of GPUs in process group
    """
     # Set up process group and gpu model
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)


def average_across_gpus(list_, device):
    """
    # Average the items in a list across all GPUs used in the process group

    Args:
        list_(list[object]): List of objects to be averaged over each GPU
        device(torch.cuda.device): Id of the device used to call this function

    Returns:
        list[object]: List averaged over each GPU
    """
    tensor = torch.tensor(list_).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM) # GLOO doesn't support AVG :(
    tensor /= dist.get_world_size()
    return tensor.tolist()
