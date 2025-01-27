'''
Author: Guido Manni
Mail: guido.manni@unicampus.it
Last Update: 07/09/23

Description:
It's the train script used to train the pose network
'''
import gc
# Python standard lib
import sys
import os
import itertools
import argparse
from tqdm import tqdm

# Numerical lib
import numpy as np

# Computer Vision lib
import cv2

# Metric lib
from evo.core import metrics
from evo.core import units
from evo.tools import file_interface

# Stat lib
import wandb

# AI-lib
import torch
from torch.optim import Adam



# Internal Module
from training_utils.utils import *
from Dataloader.cyclevodataloader import *
from training_utils import TrainingLoss, LearnableScaleConsistencyLoss
from UTILS.io_utils import ModelIO, TXTIO
from dataloader import DatasetsIO
from UTILS.geometry_utils import PoseOperator


datasetIO = DatasetsIO()
modelIO = ModelIO()
poseOperator = PoseOperator()
txtIO = TXTIO()


def train_loop(models, optimizers, train_loader, DEVICE, PaD_shape, weights_identity_loss, weights_cycle_loss, standard_identity, standard_cycle):
        # Unpack dict
        G_AB = models['G_AB'].train()
        G_BA = models['G_BA'].train()
        PaD_A = models['PaD_A'].train()
        PaD_B = models['PaD_B'].train()
        optimizer_G = optimizers['G']
        optimizer_PaD_A = optimizers['PaD_A']
        optimizer_PaD_B = optimizers['PaD_B']

        # initialize local losses for the current epoch
        loss_G_epoch = 0.0  # Initialize loss for this epoch
        loss_GAN_epoch = 0.0
        loss_identity_epoch = 0.0
        loss_D_epoch = 0.0
        loss_cycle_epoch = 0.0
        num_batches = 0

        for batch, data in enumerate(train_loader):
            real_rgb1 = data["rgb1"]
            real_rgb2 = data["rgb2"]

            real_fr1 = real_rgb1.to(DEVICE)
            real_fr2 = real_rgb2.to(DEVICE)
            stacked_frame11 = torch.cat([real_fr1, real_fr1], dim=1)
            stacked_frame22 = torch.cat([real_fr2, real_fr2], dim=1)

            disc_output = (PaD_shape.output_shape[0], 2 * PaD_shape.output_shape[1], 2 * PaD_shape.output_shape[2])
            valid = torch.Tensor(np.ones((real_fr1.size(0), *disc_output))).to(
                DEVICE)  # requires_grad = False. Default.
            fake = torch.Tensor(np.zeros((real_fr1.size(0), *disc_output))).to(
                DEVICE)  # requires_grad = False. Default.

            # Training the Generator and Pose Network

            optimizer_G.zero_grad()

            # Estimate the pose
            stacked_frame12 = torch.cat([real_fr1, real_fr2], dim=1)
            stacked_frame21 = torch.cat([real_fr2, real_fr1], dim=1)
            estimated_pose_AB_SE3 = G_AB(stacked_frame12, mode="pose")
            estimated_pose_BA_SE3 = G_BA(stacked_frame21, mode="pose")

            # Identity Loss
            identity_motion = torch.eye(4).unsqueeze(0).expand(estimated_pose_AB_SE3.shape[0], -1, -1).to(DEVICE)

            if standard_identity:
                print("standard_id")
                # we compute the standard identity loss of the cyclegan
                identity_fr1 = G_BA(stacked_frame11, identity_motion)
                identity_fr2 = G_AB(stacked_frame22, identity_motion)
                total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2, real_fr2)

            else:
                print("not standard_id")
                # we compute our custom identity loss
                identity_fr1 = G_BA(stacked_frame11, identity_motion)
                identity_fr2 = G_AB(stacked_frame22, identity_motion)
                identity_stacked_fr1 = torch.cat([identity_fr1, real_fr1], dim=1)
                identity_stacked_fr2 = torch.cat([identity_fr2, real_fr2], dim=1)
                identity_p1 = G_BA(identity_stacked_fr1, mode="pose")
                identity_p2 = G_AB(identity_stacked_fr2, mode="pose")

                total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1,
                                                                        identity_motion, identity_fr2, real_fr2,
                                                                        identity_p2, identity_motion,
                                                                        weights_identity_loss)

            # GAN loss
            fake_fr2 = G_AB(stacked_frame11, estimated_pose_AB_SE3)
            fake_fr1 = G_BA(stacked_frame22, estimated_pose_BA_SE3)
            curr_frame_fake_2 = torch.cat([fake_fr2, fake_fr2], dim=1)
            curr_frame_fake_1 = torch.cat([fake_fr1, fake_fr1], dim=1)
            loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame_fake_2, task="discriminator"), valid,
                                                      PaD_A(curr_frame_fake_1, task="discriminator"), valid)

            # Cycle loss
            if standard_cycle:
                print("standard_cycle")
                # we compute the standard cycle loss of the cyclegan
                recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
            else:
                print("not standard_cycle")
                # we compute our custom cycle loss
                recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                recov_fr12 = torch.cat([recov_fr1, recov_fr2], dim=1)
                recov_fr21 = torch.cat([recov_fr2, recov_fr1], dim=1)
                recov_P12 = G_BA(recov_fr12, mode="pose")
                recov_P21 = G_AB(recov_fr21, mode="pose")
                total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12, estimated_pose_AB_SE3,
                                                                  recov_fr2, real_fr2, recov_P21, estimated_pose_BA_SE3,
                                                                  weights_cycle_loss)

            loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss)
            loss_G.backward()
            optimizer_G.step()

            # Training the Discriminator A

            optimizer_PaD_A.zero_grad()
            prev_frame_real = torch.cat([real_fr1, real_fr1], dim=1)
            prev_frame_fake = torch.cat([fake_fr1.detach(), fake_fr1.detach()], dim=1)
            loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame_real, task='discriminator'), valid,
                                                         PaD_A(prev_frame_fake, task='discriminator'), fake)

            loss_DA.backward()
            optimizer_PaD_A.step()

            # Training the Discriminator B
            prev_frame_real = torch.cat([real_fr2, real_fr2], dim=1)
            prev_frame_fake = torch.cat([fake_fr2.detach(), fake_fr2.detach()], dim=1)
            optimizer_PaD_B.zero_grad()

            loss_DB = losses.standard_discriminator_loss(PaD_B(prev_frame_real, task='discriminator'), valid,
                                                         PaD_B(prev_frame_fake, task='discriminator'), fake)

            loss_DB.backward()
            optimizer_PaD_B.step()

            # total discriminator loss (not backwarded! -> used only for tracking)
            loss_D = (loss_DA + loss_DB) / 2

            loss_G_epoch += loss_G.item()
            loss_GAN_epoch += loss_GAN.item()
            loss_D_epoch += loss_D.item()
            loss_cycle_epoch += total_cycle_loss.item()
            loss_identity_epoch += total_identity_loss.item()
            num_batches += 1

        wandb.log({"training_loss_G": loss_G_epoch / num_batches,
                   "training_loss_GAN": loss_GAN / num_batches,
                   "training_loss_D": loss_D_epoch / num_batches,
                   "training_loss_cycle": loss_cycle_epoch / num_batches,
                   "training_loss_identity": loss_identity_epoch / num_batches,
                   })


def testing_loop(testing_root_content, models, num_worker, weights_identity_loss, weights_cycle_loss, standard_identity, standard_cycle):
    # Unpack dict
    G_AB = models['G_AB'].eval()
    G_BA = models['G_BA'].eval()
    PaD_A = models['PaD_A'].eval()
    PaD_B = models['PaD_B'].eval()

    total_testing_pose_loss = 0.0
    loss_testing_G_epoch = 0.0
    loss_testing_GAN_epoch = 0.0
    loss_testing_identity_epoch = 0.0
    loss_testing_D_epoch = 0.0
    loss_testing_cycle_epoch = 0.0
    num_batches = 0
    count = 0

    ATE_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ARE_metric = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
    RTE_metric = metrics.RPE(metrics.PoseRelation.translation_part)
    RRE_metric = metrics.RPE(metrics.PoseRelation.rotation_angle_deg)

    ground_truth_pose = []
    predictions = []
    ATE = []
    ARE = []
    RRE = []
    RTE = []
    for i in range(len(testing_root_content)):
        testing_loader = datasetIO.endoslam_dataloader(testing_root_content, batch_size=1, num_worker=num_worker, i=i)
        gt_list = [np.eye(4)]
        pd_list = [np.eye(4)]

        with torch.no_grad():
            for batch, data in enumerate(testing_loader):
                real_rgb1 = data["rgb1"]
                real_rgb2 = data["rgb2"]
                pose_fr1 = data["target"][0].to(DEVICE).float()
                pose_fr2 = data["target"][1].to(DEVICE).float()
                relative_pose = data["target"][2].to(DEVICE).float()

                real_fr1 = real_rgb1.to(DEVICE)
                real_fr2 = real_rgb2.to(DEVICE)
                stacked_frame11 = torch.cat([real_fr1, real_fr1], dim=1)
                stacked_frame22 = torch.cat([real_fr2, real_fr2], dim=1)

                # Evaluate GAN & POSE

                # Adversarial ground truths
                disc_output = (PaD_shape.output_shape[0], 2 * PaD_shape.output_shape[1], 2 * PaD_shape.output_shape[2])
                valid = torch.Tensor(np.ones((real_fr1.size(0), *disc_output))).to(
                    DEVICE)  # requires_grad = False. Default.
                fake = torch.Tensor(np.zeros((real_fr1.size(0), *disc_output))).to(
                    DEVICE)  # requires_grad = False. Default.

                # Estimate the pose
                stacked_frame12 = torch.cat([real_fr1, real_fr2], dim=1)
                stacked_frame21 = torch.cat([real_fr2, real_fr1], dim=1)
                estimated_pose_AB_SE3 = G_AB(stacked_frame12, mode="pose")
                estimated_pose_BA_SE3 = G_BA(stacked_frame21, mode="pose")

                # get the absolute poses
                pd_rel = estimated_pose_AB_SE3.squeeze().cpu().numpy()
                gt_rel = relative_pose.squeeze().cpu().numpy()
                pd_abs = pd_list[-1] @ pd_rel
                gt_abs = gt_list[-1] @ gt_rel
                # ensure the matrix is SO3 valid (needed for metric computation)
                pd_abs[:3, :3] = poseOperator.ensure_so3_v2(pd_abs[:3, :3])
                gt_abs[:3, :3] = poseOperator.ensure_so3_v2(gt_abs[:3, :3])
                gt_list.append(gt_abs)
                pd_list.append(pd_abs)

                # Identity Loss
                # identity_motion = torch.zeros(estimated_pose_AB_SE3.shape[0], estimated_pose_AB_SE3.shape[1]).to(DEVICE)
                identity_motion = torch.eye(4).unsqueeze(0).expand(estimated_pose_AB_SE3.shape[0], -1, -1).to(
                    DEVICE)

                if standard_identity:
                    # we compute the standard identity loss of the cyclegan
                    identity_fr1 = G_BA(stacked_frame11, identity_motion)
                    identity_fr2 = G_AB(stacked_frame22, identity_motion)
                    total_identity_loss = losses.standard_total_cycle_loss(identity_fr1, real_fr1, identity_fr2,
                                                                           real_fr2)

                else:
                    # we compute our custom identity loss
                    identity_fr1 = G_BA(stacked_frame11, identity_motion)
                    identity_fr2 = G_AB(stacked_frame22, identity_motion)
                    identity_stacked_fr1 = torch.cat([identity_fr1, real_fr1], dim=1)
                    identity_stacked_fr2 = torch.cat([identity_fr2, real_fr2], dim=1)
                    identity_p1 = G_BA(identity_stacked_fr1, mode="pose")
                    identity_p2 = G_AB(identity_stacked_fr2, mode="pose")

                    total_identity_loss = losses.custom_total_identity_loss(identity_fr1, real_fr1, identity_p1,
                                                                            identity_motion, identity_fr2, real_fr2,
                                                                            identity_p2, identity_motion,
                                                                            weights_identity_loss)

                # GAN loss
                fake_fr2 = G_AB(stacked_frame11, estimated_pose_AB_SE3)
                fake_fr1 = G_BA(stacked_frame22, estimated_pose_BA_SE3)
                curr_frame_fake_2 = torch.cat([fake_fr2, fake_fr2], dim=1)
                curr_frame_fake_1 = torch.cat([fake_fr1, fake_fr1], dim=1)
                loss_GAN = losses.standard_total_GAN_loss(PaD_B(curr_frame_fake_2, task="discriminator"), valid,
                                                          PaD_A(curr_frame_fake_1, task="discriminator"), valid)

                # Cycle loss
                if standard_cycle:
                    # we compute the standard cycle loss of the cyclegan
                    recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                    recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                    total_cycle_loss = losses.standard_total_cycle_loss(recov_fr1, real_fr1, recov_fr2, real_fr2)
                else:
                    # we compute our custom cycle loss
                    recov_fr1 = G_BA(curr_frame_fake_2, estimated_pose_BA_SE3)
                    recov_fr2 = G_AB(curr_frame_fake_1, estimated_pose_AB_SE3)
                    recov_fr12 = torch.cat([recov_fr1, recov_fr2], dim=1)
                    recov_fr21 = torch.cat([recov_fr2, recov_fr1], dim=1)
                    recov_P12 = G_BA(recov_fr12, mode="pose")
                    recov_P21 = G_AB(recov_fr21, mode="pose")
                    total_cycle_loss = losses.custom_total_cycle_loss(recov_fr1, real_fr1, recov_P12,
                                                                      estimated_pose_AB_SE3,
                                                                      recov_fr2, real_fr2, recov_P21,
                                                                      estimated_pose_BA_SE3,
                                                                      weights_cycle_loss)


                loss_G = loss_GAN + (10.0 * total_cycle_loss) + (5.0 * total_identity_loss)
                # Evaluate Discriminators
                prev_frame_real = torch.cat([real_fr1, real_fr1], dim=1)
                prev_frame_fake = torch.cat([fake_fr1.detach(), fake_fr1.detach()], dim=1)
                loss_DA = losses.standard_discriminator_loss(PaD_A(prev_frame_real, task='discriminator'), valid,
                                                             PaD_A(prev_frame_fake, task='discriminator'), fake)

                loss_DB = losses.standard_discriminator_loss(PaD_B(prev_frame_real, task='discriminator'), valid,
                                                             PaD_B(prev_frame_fake, task='discriminator'), fake)

                # total discriminator loss (not backwarded! -> used only for tracking)
                loss_D = (loss_DA + loss_DB) / 2

                loss_testing_G_epoch += loss_G.item()
                loss_testing_GAN_epoch += loss_GAN.item()
                loss_testing_D_epoch += loss_D.item()
                loss_testing_cycle_epoch += total_cycle_loss.item()
                loss_testing_identity_epoch += total_identity_loss.item()

                num_batches += 1

        # compute ATE, ARE, RRE, RTE
        # save trajectory in kitti format
        path_to_tmp_pose_folder = os.path.join(os.getcwd(), "tmp_test_pose")
        os.mkdir(path_to_tmp_pose_folder)
        if not os.path.exists(path_to_tmp_pose_folder):
            os.makedirs(path_to_tmp_pose_folder)
        path_to_tmp_gt = path_to_tmp_pose_folder + id + "_gt.txt"
        path_to_tmp_pd = path_to_tmp_pose_folder + id + "_pd.txt"
        save_poses_as_kitti(gt_list, path_to_tmp_gt)
        save_poses_as_kitti(pd_list, path_to_tmp_pd)

        # now load trajectories with evo
        gt_traj = file_interface.read_kitti_poses_file(path_to_tmp_gt)
        pd_traj = file_interface.read_kitti_poses_file(path_to_tmp_pd)

        # align and correct the scale
        pd_traj.align_origin(gt_traj)
        pd_traj.align(gt_traj, correct_scale=True)

        data = (gt_traj, pd_traj)
        ATE_metric.process_data(data)
        ARE_metric.process_data(data)
        RTE_metric.process_data(data)
        RRE_metric.process_data(data)

        ate = ATE_metric.get_statistic(metrics.StatisticsType.rmse)
        are = ARE_metric.get_statistic(metrics.StatisticsType.rmse)
        rte = RTE_metric.get_statistic(metrics.StatisticsType.rmse)
        rre = RRE_metric.get_statistic(metrics.StatisticsType.rmse)

        ATE.append(ate)
        ARE.append(are)
        RRE.append(rre)
        RTE.append(rte)

    ate = 0.0
    are = 0.0
    rre = 0.0
    rte = 0.0
    num_dataset = len(ATE)
    for i in range(num_dataset):
        ate += ATE[i]
        are += ARE[i]
        rre += RRE[i]
        rte += RTE[i]

    # compute the other metrics
    wandb.log({"testing_loss_G": loss_testing_G_epoch / num_batches,
               "testing_loss_GAN": loss_testing_GAN_epoch / num_batches,
               "testing_loss_D": loss_testing_D_epoch / num_batches,
               "testing_loss_cycle": loss_testing_cycle_epoch / num_batches,
               "testing_loss_identity": loss_testing_identity_epoch / num_batches,
               "ATE": ate / num_dataset,
               "ARE": are / num_dataset,
               "RRE": rre / num_dataset,
               "RTE": rte / num_dataset,
               })

    pose_metrics = {
        "ate" : ate,
        "are" : are,
        "rre" : rre,
        "rte" : rte,
    }

    return pose_metrics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--training_dataset_path", type=str, help="Path to the training dataset")
    parser.add_argument("--testing_dataset_path", type=str, help="Path to the testing dataset")
    parser.add_argument("--num_epoch", type=int, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--path_to_model", type=str, help="Path to save the trained model")
    parser.add_argument("--load_model", type=int, help="Flag to indicate whether to load a pre-trained model")
    parser.add_argument("--num_worker", type=int, default=10, help="Number of workers (default: 10)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate (default: 0.0002)")
    parser.add_argument("--betas", type=float, default=(0.5, 0.999), help="Betas (default: (0.5, 0.999))")
    parser.add_argument("--input_shape", type=int, nargs=3, default=[6, 128, 128], help="Input shape as a list (default: [3, 256, 256])")
    parser.add_argument("--standard_cycle", type=int, help="Standard flag (default: False)")
    parser.add_argument("--standard_identity", type=int, help="Standard flag (default: False)")
    parser.add_argument("--weigths_id_loss", nargs='*', type=float)
    parser.add_argument("--weigths_cycle_loss", nargs='*', type=float)
    parser.add_argument("--id", type=str)
    parser.add_argument("--id_wandb_run", type=str)

    args = parser.parse_args()

    if not os.path.exists(args.path_to_model):
        os.mkdir(args.path_to_model)

    load_model = check_value(args.load_model)
    standard_cycle = check_value(args.standard_cycle)
    standard_identity = check_value(args.standard_identity)

    # We print the training settings
    print_training_settings(args)

    # we start a new run
    wandb.init(
        project="CycleVO",
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "epoch": args.num_epoch,
        }
    )

    # Get the device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # step 1: we need to load the datasets
    training_root_content = load_InternalDT(args.training_dataset_path)
    testing_root_content = load_EndoSlam(args.testing_dataset_path)

    # step 2: get the models
    models, PaD_shape = get_models(args.input_shape, DEVICE)

    # step 3: get the optimizers
    optimizers = get_optimizers(models, args.lr, args.betas)

    # step 4: we initialize the losses
    losses = TrainingLoss()

    if load_model:
        models, optimizers, training_var = get_checkpoint(models, optimizers, args.path_to_model)

    best_metrics = {'ATE': float('inf'), 'ARE': float('inf'), 'RTE': float('inf'), 'RRE': float('inf')}

    i_folder = 0

    for epoch in tqdm(range(args.num_epoch)):
        print("[INFO]: training...")
        # get dataset
        train_loader, i_folder = internalDT_dataloader(training_root_content, batch_size=args.batch_size,
                                                           num_worker=args.num_worker, i_folder=i_folder)

        train_loop(models, optimizers, train_loader, DEVICE, PaD_shape, weights_identity_loss=args.weigths_id_loss, weights_cycle_loss=args.weigths_cycle_loss, standard_cycle=standard_cycle, standard_identity=standard_identity)


        print("[INFO]: evaluating...")
        pose_metrics = testing_loop(testing_root_content, models, args.num_worker, weights_identity_loss=args.weigths_id_loss, weights_cycle_loss=args.weigths_cycle_loss, standard_cycle=standard_cycle, standard_identity=standard_identity)

        print("[INFO]: saving models...")
        save_best_model(args.path_to_model, models, optimizers, pose_metrics, best_metrics, epoch, i_folder)










