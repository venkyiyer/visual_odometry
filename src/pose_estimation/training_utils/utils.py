from BodySLAM_Refactored.src.pose_estimation.Architecture.cycle_vo import CycleVO
from BodySLAM_Refactored.src.pose_estimation.Architecture.discriminator import Discriminator

import itertools
from torch.optim import Adam
import torch
import numpy




def check_value(value):
    assert value == 1 or value == 0, "value must be 0 or 1"
    if value == 1:
        return True
    elif value == 0:
        return False

def print_training_settings(args):
    print("-----------------------------------------------")
    print("[INFO]: Settings RECAP:")
    print("-----------------------------------------------")
    print(f"Num Epoch: {args.num_epoch}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"standard_cycle: {args.standard_cycle}")
    print(f"standard_identity: {args.standard_identity}")
    print(f"Load Model: {args.load_model}")
    print(f"weights_cycle_loss: {args.weights_cycle_loss}")
    print(f"weights_identity_loss: {args.weights_identity_loss}")
    print(f"id: {args.id}")
    print(f"wandb id run: {args.id_wandb_run}")


def get_models(input_shape, device):
    # step 2: we initialize the models
    G_AB = CycleVO(device=device, input_shape=input_shape).to(device)  # the generator that from A generates B
    G_BA = CycleVO(device=device, input_shape=input_shape).to(device)  # the generator that from B generates A
    PaD_A = Discriminator(input_shape=input_shape, device=device).to(device)  # the Pose estimator and the discriminator of A
    PaD_B = Discriminator(input_shape=input_shape, device=device).to(device)  # the Pose estimator and the discriminator of B

    PaD_shape = Discriminator(input_shape=input_shape, device=device)

    models = {
        "G_AB": G_AB,
        "G_BA": G_BA,
        "PaD_A": PaD_A,
        "PaD_B": PaD_B,
    }

    return models, PaD_shape

def get_optimizers(models, lr, betas = (0.5, 0.999)):
    optimizer_G = Adam(params=itertools.chain(models['G_AB'].parameters(), models['G_BA'].parameters()), lr=lr, betas=betas)  # optimizer for the GANs
    optimizer_PaD_A = Adam(params=models['PaD_A'].parameters(), lr=lr, betas=betas)
    optimizer_PaD_B = Adam(params=models['PaD_B'].parameters(), lr=lr, betas=betas)

    optimizers = {
        'G': optimizer_G,
        'PaD_A': optimizer_PaD_A,
        'PaD_B': optimizer_PaD_B
    }

    return optimizers


def load_model(path_to_the_model, model, optimizer = None):
    '''
    This function load the pose model.

    Parameters:
    - path_to_the_model: the path to the model [str]
    - model: pytorch model
    - optimizer: pytorch optimizer

    Return
    - model: the loaded model
    - optimizer: the loaded optimizer
    - training_var: dictionary containing the epoch, iter_on_ucbm and the best loss
    '''

    checkpoint = torch.load(path_to_the_model)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    training_var = {'epoch': checkpoint['epoch'],
                    'iter_on_ucbm': checkpoint['iter_on_ucbm'],
                    'ate': checkpoint['ate'],
                    'are': checkpoint['are'],
                    'rte': checkpoint['rte'],
                    'rre': checkpoint['rre']}

    return model, optimizer, training_var

def get_checkpoint(models, optimizers, path_to_checkpoint):
    path_to_G_AB = path_to_checkpoint + 'cyclevo_ab.pth'
    path_to_G_BA = path_to_checkpoint + 'cyclevo_ba.pth'
    path_to_PaD_A = path_to_checkpoint + "_PaD_A.pth"
    path_to_PaD_B = path_to_checkpoint + "_PaD_B.pth"

    # load G_AB
    G_AB, _, training_var = load_model(path_to_G_AB, models['G_AB'], optimizers['G'])
    # load G_BA
    G_BA, optimizer_G, _ = load_model(path_to_G_BA, models['G_BA'], optimizers['G'])
    # load PaD_A
    PaD_A, optimizer_PaD_A, _ = load_model(path_to_PaD_A, models['PaD_A'], optimizers['PaD_A'])
    # load PaD_B
    PaD_B, optimizer_PaD_B, _ = load_model(path_to_PaD_B, models['PaD_B'], optimizers['PaD_B'])

    models['G_AB'] = G_AB
    models['G_BA'] = G_BA
    models['PaD_A'] = PaD_A
    models['PaD_B'] = PaD_B

    optimizers['G'] = optimizer_G
    optimizers['PaD_A'] = optimizer_PaD_A
    optimizers['PaD_B'] = optimizer_PaD_B

    return models, optimizers, training_var

def save_poses_as_kitti(poses_list, output_path):
    """Save a list of 4x4 numpy array poses in KITTI format after ensuring SO(3) validity."""

    # Ensure that all rotation matrices are in SO(3)
    corrected_poses = []
    for pose in poses_list:
        corrected_pose = np.copy(pose)
        corrected_poses.append(corrected_pose)

    # Save the poses to a .txt file with each pose on one line
    with open(output_path, 'w') as f:
        for pose in corrected_poses:
            # Flatten the pose matrix and write as a single line
            f.write(" ".join(map(str, pose.flatten()[:-4])) + "\n")

def save_pose_model(saving_path, model, optimizer, training_var, best_model):
    '''
    This function save the pose model

    Parameters:
    - saving_path: the path where to save the model [str]
    - model: the model to save [pytorch model]
    - optimizer: the optimizer to save [pytorch optimizer]
    - training_var: dict containing some training variable to save [dict]
    - best_model: flag value to tell if it's the best model or not [bool]
    '''

    if best_model:
        curr_name = saving_path.split("/")[-1]
        new_name = curr_name.replace("model", "best_model")
        saving_path = saving_path.replace(curr_name, new_name)
    torch.save({
        'epoch': training_var['epoch'],
        'iter_on_InternalDT': training_var['iter_on_InternalDT'],
        'ate': training_var['ate'],
        'are': training_var['are'],
        'rte': training_var['rte'],
        'rre': training_var['rre'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, saving_path)

def save_best_model(path_to_model, models, optimizers, pose_metrics, best_metrics, epoch, i_folder):
    # Check if the current metrics are the best so far
    avg_pose_metrics = (pose_metrics['ate'] + pose_metrics['are'] + pose_metrics['rte'] + pose_metrics['rre']) / 4
    avg_best_pose_metrics = (best_metrics['ATE'] + best_metrics['ARE'] + best_metrics['RTE'] + best_metrics['RRE']) / 4
    if avg_pose_metrics < avg_best_pose_metrics:
        best_metrics.update({'ATE': pose_metrics['ate'], 'ARE': pose_metrics['are'], 'RTE': pose_metrics['rte'], 'RRE': pose_metrics['rre']})
        print("[INFO]: saving the best models")
        saving_path_gab = path_to_model + id + "_model_cyclevo_ab.pth"
        saving_path_gba = path_to_model + id + "_model_cyclevo_ba.pth"
        saving_path_pada = path_to_model + id + "_model_PaD_A.pth"
        saving_path_padb = path_to_model + id + "_model_PaD_B.pth"
        training_var = {'epoch': epoch,
                        'iter_on_InternalDT': i_folder,
                        'ate': pose_metrics['ate'],
                        'are': pose_metrics['are'],
                        'rre': pose_metrics['rte'],
                        'rte': pose_metrics['rre'], }
        # save generators
        save_pose_model(saving_path_gab, models['G_AB'], optimizers['G'], training_var, best_model=True)
        save_pose_model(saving_path_gba, models['G_BA'], optimizers['G'], training_var, best_model=True)
        # save discriminator model
        save_pose_model(saving_path_pada, models['PaD_A'], optimizers['PaD_A'], training_var, best_model=True)
        save_pose_model(saving_path_padb, models['PaD_B'], optimizers['PaD_B'], training_var, best_model=True)
    else:
        # we save the model as a normal one:
        print("[INFO]: saving the models")
        saving_path_gab = path_to_the_model + id + "model_cyclevo_ab.pth"
        saving_path_gba = path_to_the_model + id + "model_cyclevo_ba.pth"
        saving_path_pada = path_to_the_model + id + "model_PaD_A.pth"
        saving_path_padb = path_to_the_model + id + "model_PaD_B.pth"
        training_var = {'epoch': epoch,
                        'iter_on_InternalDT': i_folder,
                        'ate': pose_metrics['ate'],
                        'are': pose_metrics['are'],
                        'rre': pose_metrics['rte'],
                        'rte': pose_metrics['rre'], }
        # save generators
        save_pose_model(saving_path_gab, models['G_AB'], optimizers['G'], training_var, best_model=False)
        save_pose_model(saving_path_gba, models['G_BA'], optimizers['G'], training_var, best_model=False)
        # save pose and discriminator model
        save_pose_model(saving_path_pada, models['PaD_A'], optimizers['PaD_A'], training_var, best_model=False)
        save_pose_model(saving_path_padb, models['PaD_B'], optimizers['PaD_B'], training_var, best_model=False)