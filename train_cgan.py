import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from code.cgan.cgan import Pix2PixModel
from tensorboardX import SummaryWriter
from code.dataset.cgan_train_sparse import CustomDataset
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
import random

# Directory paths
root_dir1 = '../DTU_TEST/'
root_dir2 = 'outputs/'

data_transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Random horizontal and vertical flipping
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
 #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data_transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
 #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



scan_list_train = [37, 24, 40, 69, 83, 97, 105, 106, 110, 114, 118, 122]
# Dataset creation

scan_list_test = [55, 63, 65]

# Tensorboard
writer = SummaryWriter('runs/train_visualization')

class Pix2PixOptions:
    def __init__(self):
        self.isTrain = True
        self.name = "Pix2PixExperiment"
        self.gpu_ids = get_available_gpus()
        self.checkpoints_dir = "./checkpoints"
        self.model = "pix2pix"
        self.input_nc = 3
        self.output_nc = 3
        self.ngf = 32
        self.ndf = 32
        self.netD = "n_layers"
        self.n_layers_D = 3
        self.netG = "local"
        self.norm = "instance"
        self.init_type = "normal"
        self.init_gain = 0.02
        self.no_dropout = True
        self.dataset_mode = "unaligned"
        self.direction = "AtoB"
        self.serial_batches = False
        self.num_threads = 4
        self.batch_size = 32
        self.load_size = 256
        self.crop_size = 256
        self.max_dataset_size = float("inf")
        self.preprocess = 'resize_and_crop'
        self.no_flip = False
        self.display_winsize = 256
        self.epoch = 'latest'
        self.load_iter = 0
        self.verbose = False
        self.suffix = ''
        self.use_wandb = False
        self.wandb_project_name = 'CycleGAN-and-pix2pix'
        self.gan_mode = 'lsgan'
        self.lr = 2e-4
        self.beta1 = 0.5
        
        self.niter = 100
        self.niter_decay = 100
        
        #multiscale_d opts
        self.use_sigmoid=False
        self.num_D = 3
        self.getIntermFeat =False
        
        self.no_lsgan = False
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        
        self.niter_fix_global = 20
        
        
        self.resize_or_crop = 'resize'
        self.loadSize = 800
        self.fineSize = 512
        # Default arguments from TrainOptions
        self.display_freq = 400
        self.display_ncols = 4
        self.display_id = 1
        self.display_server = "http://localhost"
        self.display_env = 'main'
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False
        self.save_latest_freq = 5000
        self.save_epoch_freq = 5
        self.save_by_iter = False
        self.continue_train = False
        self.epoch_count = 1
        self.phase = 'train'
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.pool_size = 50
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.lambda_L1 = 10.0
        self.lambda_L2 = 0.0
        self.lambda_VGG = 10.0
        
        self.gen_features = False

def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]

# Model instantiation

pix2pix_options = Pix2PixOptions()
pix2pix_model = Pix2PixModel(pix2pix_options)

# Training loop
num_epochs = 10000  # Define your number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DataLoader for test dataset
dataset_train = CustomDataset(root_dir1, root_dir2, scan_list_train, opt=pix2pix_options, isTrain=True)
dataset_test = CustomDataset(root_dir1, root_dir2, scan_list_test, opt=pix2pix_options, isTrain=False)

# DataLoader
batch_size = 2  # Define your batch size
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)

for epoch in range(num_epochs):
    # Train
    for i, data in enumerate(data_loader):
        real_A = data['input'].to(device)
        real_B = data['ground_truth'].to(device)

        pix2pix_model.set_input({"A": real_A, "B": real_B})
        pix2pix_model.optimize_parameters()

        if i % 2 == 0:
            pix2pix_model.forward()
            fake_B = pix2pix_model.fake_B.detach()

            # Reverse normalization for visualization
            real_A_vis = real_A.clone().cpu().data
            real_A_vis = real_A_vis * 0.5 + 0.5
            fake_B_vis = fake_B.clone().cpu().data
            fake_B_vis = fake_B_vis * 0.5 + 0.5
            real_B_vis = real_B.clone().cpu().data
            real_B_vis = real_B_vis * 0.5 + 0.5

            writer.add_scalar('Train/Epoch_{}/Generator Loss (VGG)', pix2pix_model.loss_G_VGG, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (MSE)', pix2pix_model.loss_G_L2, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (L1)', pix2pix_model.loss_G_L1, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss', pix2pix_model.loss_G, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Fake Loss', pix2pix_model.loss_D_fake, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Real Loss', pix2pix_model.loss_D_real, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Discriminator Loss', pix2pix_model.loss_D, global_step=epoch * len(data_loader) + i)

            img_grid = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
            img_grid = make_grid(img_grid, nrow=3)

            writer.add_image('Train/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)


    # Test/Evaluate
    with torch.no_grad():  # do not need to compute gradients when evaluating
        for i, data in enumerate(data_loader_test):
            real_A = data['input'].to(device)
            real_B = data['ground_truth'].to(device)

            pix2pix_model.set_input({"A": real_A, "B": real_B})
            pix2pix_model.forward()
            fake_B = pix2pix_model.fake_B.detach()

            # Reverse normalization for visualization
            real_A_vis = real_A.clone().cpu().data
            real_A_vis = real_A_vis * 0.5 + 0.5
            fake_B_vis = fake_B.clone().cpu().data
            fake_B_vis = fake_B_vis * 0.5 + 0.5
            real_B_vis = real_B.clone().cpu().data
            real_B_vis = real_B_vis * 0.5 + 0.5
            
            pix2pix_model.backward_D(True)
            pix2pix_model.backward_G(True)

            writer.add_scalar('Test/Epoch_{}/Generator Loss', pix2pix_model.loss_G, global_step=epoch * len(data_loader_test) + i)
            writer.add_scalar('Test/Epoch_{}/Discriminator Loss', pix2pix_model.loss_D, global_step=epoch * len(data_loader_test) + i)

            img_grid = torch.cat((real_A_vis, fake_B_vis, real_B_vis), 2)
            img_grid = make_grid(img_grid, nrow=3)

            writer.add_image('Test/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)
    #if epoch > pix2pix_options.niter:
    #    pix2pix_model.update_learning_rate()

    if (pix2pix_options.niter_fix_global != 0) and (epoch == pix2pix_options.niter_fix_global):
        pix2pix_model.update_fixed_params()

    print(f'Epoch {epoch+1}/{num_epochs} finished.')
