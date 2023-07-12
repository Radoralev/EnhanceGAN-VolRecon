import torch
import argparse
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader
from code.cgan.cgan import Pix2PixModel
from tensorboardX import SummaryWriter
from code.dataset.cgan_train_sparse import CGanTrainDataset
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.spatial.distance import cdist
from torchvision.transforms import ToPILImage
from cgan_options import Pix2PixOptions
import cv2


def calculate_metrics(real_A, real_B, fake_B, device):
    # Convert tensors to numpy arrays for metric calculation
    real_A_np = real_A.cpu().numpy().squeeze()
    real_B_np = real_B.cpu().numpy().squeeze()
    fake_B_np = fake_B.cpu().numpy().squeeze()

    real_A_np = real_A_np.transpose((1, 2, 0))
    real_B_np = real_B_np.transpose((1, 2, 0))
    fake_B_np = fake_B_np.transpose((1, 2, 0))

    # Now compute the metrics
    psnr_input_gt = peak_signal_noise_ratio(real_A_np, real_B_np, data_range=1)
    psnr_gt_output = peak_signal_noise_ratio(
        real_B_np, fake_B_np, data_range=1)
    ssim_input_gt = structural_similarity(
        real_A_np, real_B_np, multichannel=True, data_range=1)
    ssim_gt_output = structural_similarity(
        real_B_np, fake_B_np, multichannel=True, data_range=1)

    # MSE
    mse_input_gt = np.mean((real_A_np - real_B_np) ** 2)
    mse_gt_output = np.mean((real_B_np - fake_B_np) ** 2)

    return {
        'psnr_input_gt': psnr_input_gt,
        'psnr_gt_output': psnr_gt_output,
        'mse_input_gt': mse_input_gt,
        'mse_gt_output': mse_gt_output,
        'ssim_input_gt': ssim_input_gt,
        'ssim_gt_output': ssim_gt_output,
    }


def load_model(pix2pix_options, ckpt_netG, ckpt_netD):
    pix2pix_model = Pix2PixModel(pix2pix_options)

    pix2pix_model.netG.load_state_dict(torch.load(ckpt_netG))
    pix2pix_model.netD.load_state_dict(torch.load(ckpt_netD))

    return pix2pix_model


def main(ckpt_netG, ckpt_netD, out):
    root_dir2 = '../Rectified_colmap/'
    root_dir1 = 'outputs_g_final/'
    # Initiate dataframe to store the metrics
    df_metrics = pd.DataFrame(columns=['Image',  # 'cdist_input_gt', 'cdist_gt_output',
                                       'psnr_input_gt', 'psnr_gt_output',
                                       'mse_input_gt', 'mse_gt_output',
                                       'ssim_input_gt', 'ssim_gt_output'])

    run_name = f'test_{time.asctime()}'
    writer = SummaryWriter(f'runs/{run_name}')

    pix2pix_options = Pix2PixOptions()

    # Load Model
    pix2pix_model = load_model(pix2pix_options, ckpt_netG, ckpt_netD)
    num_parameters_D = sum(
        p.numel() for p in pix2pix_model.netD.parameters() if p.requires_grad)
    num_parameters_G = sum(
        p.numel() for p in pix2pix_model.netG.parameters() if p.requires_grad)

    print('Number of parameters in discriminator: ', num_parameters_D)
    print('Number of parameters in generator: ', num_parameters_G)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = CGanTrainDataset(
        root_dir1, root_dir2, opt=pix2pix_options, isTrain=False)
    data_loader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=8)

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader_test)):
            real_A = data['input'].to(device)
            real_B = data['ground_truth'].to(device)
            depth_A = data['depth_image'].to(device)

            real_A_depth = torch.cat((real_A, depth_A), dim=1)
            pix2pix_model.set_input({"A": real_A_depth, "B": real_B})
            pix2pix_model.forward()

            fake_B = pix2pix_model.fake_B.detach()

            real_A_vis = real_A.clone().cpu().data
            real_A_vis = real_A_vis * 0.5 + 0.5
            fake_B_vis = fake_B.clone().cpu().data
            fake_B_vis = fake_B_vis * 0.5 + 0.5
            real_B_vis = real_B.clone().cpu().data
            real_B_vis = real_B_vis * 0.5 + 0.5

            metrics = calculate_metrics(
                real_A_vis, real_B_vis, fake_B_vis, device)

            # Add to dataframe
            df_metrics.loc[i] = ['Image_{}'.format(i)] + list(metrics.values())

            # Log metrics to TensorBoard
            for metric, value in metrics.items():
                writer.add_scalar(f'Test/{metric}', value, global_step=i)

            img_grid = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
            img_grid = make_grid(img_grid, nrow=3)
            writer.add_image('Test/Image_{}'.format(i),
                             img_grid, global_step=i)

    df_metrics['psnr_input_gt'] = df_metrics['psnr_input_gt'].mean()
    df_metrics['psnr_gt_output'] = df_metrics['psnr_gt_output'].mean()
    df_metrics['mse_input_gt'] = df_metrics['mse_input_gt'].mean()
    df_metrics['mse_gt_output'] = df_metrics['mse_gt_output'].mean()
    df_metrics['ssim_input_gt'] = df_metrics['ssim_input_gt'].mean()
    df_metrics['ssim_gt_output'] = df_metrics['ssim_gt_output'].mean()
    df_metrics = df_metrics.drop_duplicates(subset=['psnr_input_gt'])
    df_metrics.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pix2Pix Evaluation')
    parser.add_argument('--ckpt_netG', type=str, required=True,
                        help='Checkpoint for generator network')
    parser.add_argument('--ckpt_netD', type=str, required=True,
                        help='Checkpoint for discriminator network')
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    main(args.ckpt_netG, args.ckpt_netD, args.out)
