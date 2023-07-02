import torch
import time
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from code.cgan.cgan import Pix2PixModel
from code.dataset.cgan_train_sparse import CGanTrainDataset
from cgan_options import Pix2PixOptions
# Directory paths
import argparse
from evaluate_cgan import calculate_metrics
import pandas as pd

root_dir2 = '../Rectified_colmap/'
root_dir1 = 'outputs_g_final/'
# Create the dataset

parser = argparse.ArgumentParser(description='Pix2Pix Evaluation')
parser.add_argument('--ckpt_netG', type=str, required=False, help='Checkpoint for generator network')
parser.add_argument('--ckpt_netD', type=str, required=False, help='Checkpoint for discriminator network')
parser.add_argument('--run_name', type=str, required=False, help='Run name')
parser.add_argument('--epoch_old', type=int, required=False, help='Epoch of checkpoint (necessary for logging).')
args = parser.parse_args()

def load_model(pix2pix_options, ckpt_netG, ckpt_netD):
    pix2pix_model = Pix2PixModel(pix2pix_options)
    state_dict_netG = torch.load(ckpt_netG)
    #print(state_dict_netG)
    pix2pix_model.netG.load_state_dict(torch.load(ckpt_netG))
    pix2pix_model.netD.load_state_dict(torch.load(ckpt_netD))
    
    return pix2pix_model


# Tensorboard
if args.run_name:
    run_name = args.run_name
else:
    run_name = f'train_{time.asctime()}'
writer = SummaryWriter(f'runs/{run_name}')


# Model instantiation
pix2pix_options = Pix2PixOptions()
if args.ckpt_netG and args.ckpt_netD:
    pix2pix_model = load_model(pix2pix_options, args.ckpt_netG, args.ckpt_netD)
else:
    pix2pix_model = Pix2PixModel(pix2pix_options)

# Training loop
num_epochs = 10000  # Define your number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DataLoader for test dataset
dataset_train = CGanTrainDataset(root_dir1, root_dir2, opt=pix2pix_options, isTrain=True)
dataset_test = CGanTrainDataset(root_dir1, root_dir2, opt=pix2pix_options, isTrain=False)

#dataset_train = CustomDataset(root_dir1, root_dir2, scan_list_train, opt=pix2pix_options, isTrain=True)
#dataset_test = CustomDataset(root_dir1, root_dir2, scan_list_test, opt=pix2pix_options, isTrain=False)

# DataLoader
batch_size = 3  # Define your batch size
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

for epoch in range(0 if not args.epoch_old else int(args.epoch_old), num_epochs):
    df_metrics = pd.DataFrame(columns=['Image', #'cdist_input_gt', 'cdist_gt_output',
                                       'psnr_input_gt', 'psnr_gt_output', 
                                       'mse_input_gt', 'mse_gt_output', 
                                       'ssim_input_gt', 'ssim_gt_output'])
    # Train
    for i, data in tqdm(enumerate(data_loader)):
        real_A = data['input'].to(device)
        real_B = data['ground_truth'].to(device)
        depth_A = data['depth_image'].to(device)

        real_A_depth = torch.cat((real_A, depth_A), dim=1)
        pix2pix_model.set_input({"A": real_A_depth, "B": real_B})
        pix2pix_model.optimize_parameters()

        if i % 28 == 0:
            pix2pix_model.forward()
            fake_B = pix2pix_model.fake_B.detach()
            
            writer.add_scalar('Train/Epoch_{}/Generator Loss (VGG)', pix2pix_model.loss_G_VGG, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (MSE)', pix2pix_model.loss_G_GAN_Feat, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (L1)', pix2pix_model.loss_G_L1, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss', pix2pix_model.loss_G, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Fake Loss', pix2pix_model.loss_D_fake, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Real Loss', pix2pix_model.loss_D_real, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Discriminator Loss', pix2pix_model.loss_D, global_step=epoch * len(data_loader) + i)

            # Reverse normalization for visualization
            real_A_vis = real_A.clone().cpu().data
            real_A_vis = real_A_vis * 0.5 + 0.5
            fake_B_vis = fake_B.clone().cpu().data
            fake_B_vis = fake_B_vis * 0.5 + 0.5
            real_B_vis = real_B.clone().cpu().data
            real_B_vis = real_B_vis * 0.5 + 0.5
            img_grid = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
            img_grid = make_grid(img_grid, nrow=3)
            writer.add_image('Train/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)

    #if (pix2pix_options.niter_fix_global != 0) and (epoch == pix2pix_options.niter_fix_global):
    #    pix2pix_model.update_fixed_params()

    # Test/Evaluate
    with torch.no_grad():  # do not need to compute gradients when evaluating
        for i, data in tqdm(enumerate(data_loader_test)):
            real_A = data['input'].to(device)
            real_B = data['ground_truth'].to(device)
            depth_A = data['depth_image'].to(device)

            real_A_depth = torch.cat((real_A, depth_A), dim=1)
            pix2pix_model.set_input({"A": real_A_depth, "B": real_B})
            pix2pix_model.forward()
            fake_B = pix2pix_model.fake_B.detach()

            
            pix2pix_model.backward_D(True)
            pix2pix_model.backward_G(True)

            writer.add_scalar('Test/Epoch_{}/Generator Loss', pix2pix_model.loss_G, global_step=epoch * len(data_loader_test) + i)
            writer.add_scalar('Test/Epoch_{}/Discriminator Loss', pix2pix_model.loss_D, global_step=epoch * len(data_loader_test) + i)

            # Reverse normalization for visualization
            real_A_vis = real_A.clone().cpu().data
            real_A_vis = real_A_vis * 0.5 + 0.5
            fake_B_vis = fake_B.clone().cpu().data
            fake_B_vis = fake_B_vis * 0.5 + 0.5
            real_B_vis = real_B.clone().cpu().data
            real_B_vis = real_B_vis * 0.5 + 0.5
            
            if (epoch+1) % 5 == 0:
                img_grid = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
                img_grid = make_grid(img_grid, nrow=3)
                writer.add_image('Test/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)
                
            metrics = calculate_metrics(real_A_vis, real_B_vis, fake_B_vis, device)
            # Add to dataframe
            df_metrics.loc[i] = ['Image_{}'.format(i)] + list(metrics.values())
            
        df_metrics['psnr_input_gt'] = df_metrics['psnr_input_gt'].mean()
        df_metrics['psnr_gt_output'] = df_metrics['psnr_gt_output'].mean()
        df_metrics['mse_input_gt'] = df_metrics['mse_input_gt'].mean()
        df_metrics['mse_gt_output'] = df_metrics['mse_gt_output'].mean()
        df_metrics['ssim_input_gt'] = df_metrics['ssim_input_gt'].mean()
        df_metrics['ssim_gt_output'] = df_metrics['ssim_gt_output'].mean()

        print(df_metrics.drop_duplicates(subset=['psnr_input_gt']))
                
        torch.save(pix2pix_model.netD.state_dict(), f'checkpoints_cgan//checkpoint_{epoch+1}_netD.pth')
        torch.save(pix2pix_model.netG.state_dict(), f'checkpoints_cgan//checkpoint_{epoch+1}_netG.pth')
        print(f'Checkpoint saved at epoch {epoch+1}')

    #if epoch > pix2pix_options.niter:
    #    pix2pix_model.update_learning_rate()


    print(f'Epoch {epoch+1}/{num_epochs} finished.')
