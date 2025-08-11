import torch
import numpy as np

from models import init_model


def ensemble_prediction_average(base_models_outputs, device):
    """
    Calculate the average prediction of base models based on average prediction probability.

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (num_models, num_images, num_classes)
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """
  
    # Get all model prediction probabilites for all imags
    probs = torch.zeros(base_models_outputs.size()).to(device) # shape (num_models, num_images, num_classes)
    for i, output in enumerate(base_models_outputs):
        probs[i] = torch.softmax(output, dim=1)

    # Return the class with highest average prediction probability across all models
    average_probs = torch.mean(probs, dim=0)
    return torch.argmax(average_probs, dim=1)


def ensemble_prediction_majority(base_models_outputs, num_images, device):
    """
    Calculate the majority prediction of base models.
    (Uses highest average probability to settle no majority cases e.g. 2 versus 2)

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (number of base models, number of images, number of classes)
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """
    
    # Get all model predictions and probabilites for all images
    preds = torch.zeros(np.shape(np.array(base_models_outputs))[:2]).to(device) # shape (num_models, num_images)
    probs = torch.zeros(np.shape(np.array(base_models_outputs))).to(device) # shape (num_models, num_images, num_classes)
    for i, output in enumerate(base_models_outputs):
        preds[i] = torch.argmax(output, dim=1)
        probs[i] = torch.softmax(output, dim=1)
    
    # One prediction per image
    out_preds = torch.zeros(np.shape(np.array(base_models_outputs))[1]).to(device) # shape (num_models, num_images)

    # Get the majority vote for each model
    for i in range(num_images):
        # Get image predictions and probabilities
        image_preds = preds[:, i] # shape (num_models)
        image_probs = probs[:, i, :] # shape (num_models, num_classes)

         # Get all the classes predicted for this model
        image_prediction, counts = image_preds.unique(return_counts=True)

        # Get the class(es) with the highest number of predictions
        majority = counts.max()
        conflicting_classes = image_prediction[counts == majority]

        # If there are multiple classes with highest count, then no majority so resolve
        if len(conflicting_classes) != 1: 

            # Get average probability for all the models that predicted this class
            average_probs = [torch.mean(image_probs[:, int(class_.item())], dim=0) for class_ in conflicting_classes]
            average_probs = torch.stack(average_probs)

            # Set image prediction to the class with the highest probability over all the images
            out_preds[i] = conflicting_classes[torch.argmax(average_probs)]
        else:
            # Otherwise set output prediction to majority vote
            out_preds[i] = image_prediction[counts == majority]

    return out_preds


def ensemble_stacking(base_models_outputs, meta_learner, device):
    """
    Calculate the meta-learner prediction based on base model outputs.

    Args:
        base_models_outputs (torch.tensor): base model outputs shape (number of base models, number of images, number of classes)
        meta_learner (nn.Module): ensemble meta-learner
        device (torch.device): device to put tensors on

    Returns:
        torch.tensor: final predictions shape (number of images)
    """

    # Concatenate base model outputs
    base_models_outputs = torch.cat(base_models_outputs, dim=1) 
    base_models_outputs = base_models_outputs.to(device)

    # Pass the outputs through the stacking model
    meta_learner = meta_learner.to(device)
    outputs = meta_learner(base_models_outputs)

    # Return the meta learner predictions for each image
    return torch.argmax(outputs, dim=1)


def get_ensemble(ensemble_config, num_classes, device):
    """
    Initialise the ensemble using the specified configuration.

    Args:
        ensemble_config (dict): Ensemble configuration file
        num_classes (int): Number of dataset classes
        device (torch.device): Device to put models on
        num_classes (int): Number of classes in the dataset

    Returns:
        torch.Module: Loaded ensemble model.
    """

    base_models = {}
    base_models_transforms = {}

    # Load each base model and their transforms
    for base_model_config in ensemble_config['base_models']:

        base_model, base_model_transform = init_model(base_model_config, num_classes, device)
        base_model.load_state_dict(torch.load(base_model_config['trained_model_path'], map_location=device), strict=False)

        base_model.eval()

        base_models[base_model_config['name']] = base_model
        base_models_transforms[base_model_config['name']] = base_model_transform


    # Initialise the meta-learner if using stacking
    if ensemble_config['ensemble_mode'] == 'stacking':

        meta_learner, _ = init_model(ensemble_config, num_classes, device)
        
        # Load from pre-trained if required
        if 'meta_learner_path' in ensemble_config:
            meta_learner.load_state_dict(torch.load(ensemble_config['meta_learner_path'], map_location=device))

        meta_learner.eval()

        return meta_learner, base_models, base_models_transforms
    
    return None, base_models, base_models_transforms