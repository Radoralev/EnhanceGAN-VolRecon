import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from code.cgan.cgan import Pix2PixModel
from tensorboardX import SummaryWriter
from code.dataset.cgan_train_sparse import CustomDataset
from torchvision.utils import make_grid

# Directory paths
root_dir1 = '../DTU_TEST/'
root_dir2 = 'outputs/'

# Dataset preparation
data_transform_train = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # Random horizontal and vertical flipping
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
 #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


data_transform_test = transforms.Compose([
    transforms.Resize((600, 800)),
    transforms.ToTensor(),
 #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

scan_list_train = [24, 37, 40, 69, 83, 97, 105, 106, 110, 114, 118, 122]
# Dataset creation
dataset_train = CustomDataset(root_dir1, root_dir2, scan_list_train, transform=data_transform_train)

scan_list_test = [55, 63, 65]
dataset_test = CustomDataset(root_dir1, root_dir2, scan_list_test, transform=data_transform_test)


# DataLoader
batch_size = 4  # Define your batch size
data_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

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
        self.ngf = 128
        self.ndf = 128
        self.netD = "n_layers"
        self.n_layers_D = 3
        self.netG = "unet_128"
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
        self.gan_mode = 'wgangp'
        self.lr = 2e-4
        self.beta1 = 0.5
        
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
        self.lambda_L1 = 100.0
        self.lambda_L2 = 0.0
        self.lambda_VGG = 1.0

def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]

# Model instantiation

pix2pix_options = Pix2PixOptions()
pix2pix_model = Pix2PixModel(pix2pix_options)


# Training loop
num_epochs = 250  # Define your number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DataLoader for test dataset
data_loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)


# Define your patch size and step
patch_size_w = 128
patch_size_h = 128
step_w = 100
step_h = 100
patch_batch_size = 32

for epoch in range(num_epochs):
    # Train
    for i, data in enumerate(data_loader):
        real_A_full = data['input'].to(device)
        real_B_full = data['ground_truth'].to(device)
        num_patches_h = (real_A_full.shape[2] - patch_size_h) // step_h + 1
        num_patches_w = (real_A_full.shape[3] - patch_size_w) // step_w + 1

        # Prepare tensors for full generated image and loss accumulation
        fake_B_full = torch.zeros_like(real_A_full)
        loss_G_accumulated = 0
        loss_D_accumulated = 0

        # Get patches
        patches_A = []
        patches_B = []
        for h in range(0, real_A_full.shape[2] - patch_size_h, step_h):
            for w in range(0, real_A_full.shape[3] - patch_size_w, step_w):
                real_A = real_A_full[..., h:h+patch_size_h, w:w+patch_size_w]
                real_B = real_B_full[..., h:h+patch_size_h, w:w+patch_size_w]
                patches_A.append(real_A)
                patches_B.append(real_B)

        # Rearrange patches into a batch
        patches_A = torch.stack(patches_A, dim=0)
        patches_B = torch.stack(patches_B, dim=0)

        # Iterate over patches in batches
        for b in range(0, len(patches_A), patch_batch_size):
            batch_A = patches_A[b:b+patch_batch_size]
            batch_B = patches_B[b:b+patch_batch_size]
        
            # Flatten the first two dimensions
            batch_A_flattened = batch_A.view(-1, *batch_A.shape[2:])
            batch_B_flattened = batch_B.view(-1, *batch_B.shape[2:])
        
            pix2pix_model.set_input({"A": batch_A_flattened, "B": batch_B_flattened})
            pix2pix_model.optimize_parameters()
        
            # Generate patches and add to full generated image
            pix2pix_model.forward()
            fake_B_batch = pix2pix_model.fake_B.detach()
        
            # Unflatten the first dimension and add patches to the full generated image
            fake_B_batch = fake_B_batch.view(*batch_A.shape)
            for p in range(b, min(b+patch_batch_size, len(patches_A))):
                h = (p // num_patches_w) * step_h
                w = (p % num_patches_w) * step_w
                fake_B_full[..., h:h+patch_size_h, w:w+patch_size_w] = fake_B_batch[p-b]
        
            # Accumulate losses
            loss_G_accumulated += pix2pix_model.loss_G
            loss_D_accumulated += pix2pix_model.loss_D

        if i % 2 == 0:  # Log every 2nd batch
            writer.add_scalar('Train/Epoch_{}/Generator Loss (VGG)', pix2pix_model.loss_G_VGG, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (MSE)', pix2pix_model.loss_G_L2, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss (L1)', pix2pix_model.loss_G_L1, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Generator Loss', loss_G_accumulated, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Fake Loss', pix2pix_model.loss_D_fake, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/D-Real Loss', pix2pix_model.loss_D_real, global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Train/Epoch_{}/Discriminator Loss', loss_D_accumulated, global_step=epoch * len(data_loader) + i)

            # Visualize the full-sized images, not patches
            img_grid = torch.cat((real_A_full, fake_B_full, real_B_full), 2)
            img_grid = make_grid(img_grid, nrow=3)

            writer.add_image('Train/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)

    print(f'Epoch {epoch+1}/{num_epochs} finished.')
    
    # Test/Evaluate
    with torch.no_grad():  # do not need to compute gradients when evaluating
        for i, data in enumerate(data_loader_test):
            real_A_full = data['input'].to(device)
            real_B_full = data['ground_truth'].to(device)

            num_patches_h = (real_A_full.shape[2] - patch_size_h) // step_h + 1
            num_patches_w = (real_A_full.shape[3] - patch_size_w) // step_w + 1

            # Prepare tensor for full generated image and loss accumulation
            fake_B_full = torch.zeros_like(real_A_full)
            loss_G_accumulated = 0
            loss_D_accumulated = 0

            # Get patches
            patches_A = []
            patches_B = []
            for h in range(0, real_A_full.shape[2] - patch_size_h, step_h):
                for w in range(0, real_A_full.shape[3] - patch_size_w, step_w):
                    real_A = real_A_full[..., h:h+patch_size_h, w:w+patch_size_w]
                    real_B = real_B_full[..., h:h+patch_size_h, w:w+patch_size_w]
                    patches_A.append(real_A)
                    patches_B.append(real_B)

            # Rearrange patches into a batch
            patches_A = torch.stack(patches_A, dim=0)
            patches_B = torch.stack(patches_B, dim=0)

            # Iterate over patches
            for b in range(0, len(patches_A)):
                real_A = patches_A[b]
                real_B = patches_B[b]

                pix2pix_model.set_input({"A": real_A, "B": real_B})
                pix2pix_model.forward()
                fake_B = pix2pix_model.fake_B.detach()

                # Add patch to the full generated image
                h = (b // num_patches_w) * step_h
                w = (b % num_patches_w) * step_w
                fake_B_full[..., h:h+patch_size_h, w:w+patch_size_w] = fake_B

                # Accumulate losses
                loss_G_accumulated += pix2pix_model.loss_G
                loss_D_accumulated += pix2pix_model.loss_D

            writer.add_scalar('Test/Epoch_{}/Generator Loss', loss_G_accumulated, global_step=epoch * len(data_loader_test) + i)
            writer.add_scalar('Test/Epoch_{}/Discriminator Loss', loss_D_accumulated, global_step=epoch * len(data_loader_test) + i)

            img_grid = torch.cat((real_A_full, fake_B_full, real_B_full), 2)
            img_grid = make_grid(img_grid, nrow=3)

            writer.add_image('Test/Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)
