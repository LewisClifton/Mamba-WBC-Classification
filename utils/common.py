import os
from datetime import datetime
import yaml

import torch
import torch.distributed as dist


def save_log(out_dir, metrics, model_config, dataset_config):
    """
    Save training log

    Args:
        out_dir(string): Path to directory to save the log file to
        date(string): Date/time of training
        metrics(dict or list[dict]): Model or list of metrics dictionarys to be saved
    """
    
    log_path = os.path.join(out_dir, 'log.txt')
    with open(log_path , 'w') as file:
        file.write(f'Date/time of creation: {datetime.now()}\n')

        if isinstance(metrics, list):
            # Save metrics for each fold
            for idx, fold_metrics in enumerate(metrics):
                file.write(f'\nFold {idx+1} training metrics:\n')
                write_dict_to_file(file, fold_metrics)
        else:
            # Save metrics for the model
            file.write(f'\nMetrics:\n')
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
        if isinstance(v, list) and len(v) == 0: continue # avoids error
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


def save_models(out_dir, trained, model_type, metrics, fold=None):
    """
    Save trained model(s) to a given output directory

    Args:
        out_dir(string): Path to directory to save the model(s) to
        trained(torch.nn.Module or list[torch.nn.Module]): Model or list of models to be saved
        using_dist(bool): Whether multiple GPUs were used to train the model(s)
    """

    if fold is not None:
        model_path = os.path.join(out_dir, f'{model_type}_fold_{fold}_acc_{metrics['Best validation accuracy during training']}.pth')
    else:
        model_path = os.path.join(out_dir, f'{model_type}_acc_{metrics['Best validation accuracy during training']}.pth')

    torch.save(trained.state_dict(), model_path)
    print(f'Saved trained model to {model_path}')

    # # Create trained models directory if needed
    # if isinstance(trained, list):
    #     # Save trained models for each fold
    #     for idx, model in enumerate(trained):
    #         model_path = os.path.join(out_dir, f'{model_type}_fold_{idx}_acc_{metrics[idx]['Best validation accuracy during training']}.pth')
    #         torch.save(model.state_dict(), model_path)
    #     print(f'\n{len(trained)} trained models saved to {out_dir}')
    # else:
    #     # Save the model
    #     model_path = os.path.join(out_dir, f'{model_type}_acc_{metrics['Best validation accuracy during training']}.pth')
    #     torch.save(trained.state_dict(), model_path)
    #     print(f'Saved trained model to {model_path}')


def save_config(out_dir, config):
    """
    # Save model config file

    Args:
        out_dir(string): Path to directory to save the config file to
        config(dict): Training config dictionary to be saved
    """

    # Save model configuration
    config_path = os.path.join(out_dir, 'config.yml')
    with open(config_path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)


def save(out_dir, metrics, trained, model_config, dataset_config):
    """
    Save trained models, training log and model configuration file

    Args:
        out_dir(string): Path to directory to save the log, models and config file to
        metrics(dict or list[dict]): Model or list of metrics dictionarys to be saved
        trained(torch.nn.Module or list[torch.nn.Module]): Model or list of models to be saved
        config(dict): Model config dictionary to be saved
        using_dist(bool): Whether multiple GPUs were used to train the model(s)
    """
    
    # Save models, log and config yml
    # save_models(out_dir, trained, model_config['name'], model_config['epochs'], metrics)
    save_log(out_dir, metrics, model_config, dataset_config)
    save_config(out_dir, model_config)