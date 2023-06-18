import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from code.cgan_hd.pix2pixHD_model import Pix2PixHDModel
from code.cgan_hd.options import TrainOptions
from tensorboardX import SummaryWriter
from code.dataset.cgan_train_sparse import CustomDataset
from torchvision.utils import make_grid

# Directory paths
root_dir1 = '../DTU_TEST/'
root_dir2 = 'outputs/'

# Dataset preparation
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
 #   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset creation
dataset = CustomDataset(root_dir1, root_dir2, transform=data_transform)

# DataLoader
batch_size = 8  # Define your batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Tensorboard
writer = SummaryWriter('runs/train_visualization')

def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]

# Model instantiation

opt = TrainOptions()


opt.isTrain = True

model = Pix2PixHDModel()
model.initialize(opt)

# Training loop
num_epochs = 150  # Define your number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


for epoch in range(num_epochs):
    epoch_start_time = time.time()
    for i, data in enumerate(data_loader):
        # Whether to collect output images
        save_fake = i % opt.display_freq == 0

        # set input data to model
        label = data['input'].to(device)
        image = data['ground_truth'].to(device)
        inst = feat = None   # Set these as per your data_loader if available
        
        # forward pass and calculate losses
        losses, generated = model.forward(label, inst, image, feat, infer=save_fake)

        # sum per device losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))

        # calculate final loss scalar
        loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
        loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0)

        # update generator weights
        model.optimizer_G.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_G, model.optimizer_G) as scaled_loss: scaled_loss.backward()                
        else:
            loss_G.backward()          
        model.optimizer_G.step()

        # update discriminator weights
        model.optimizer_D.zero_grad()
        if opt.fp16:                                
            with amp.scale_loss(loss_D, model.optimizer_D) as scaled_loss: scaled_loss.backward()                
        else:
            loss_D.backward()        
        model.optimizer_D.step()     

        # Tensorboard visualization
        if i % 2 == 0:  # adjust this condition to control how often you want to log images
            # Log the generator and discriminator losses
            writer.add_scalar('Generator Loss', loss_G.item(), global_step=epoch * len(data_loader) + i)
            writer.add_scalar('Discriminator Loss', loss_D.item(), global_step=epoch * len(data_loader) + i)
            
            # create grid of images
            visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
            img_grid = make_grid([visuals['input_label'], visuals['synthesized_image'], visuals['real_image']])
            
            # add to tensorboard writer
            writer.add_image('Epoch_{}/Image_{}'.format(epoch, i), img_grid, global_step=epoch)

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        model.module.update_learning_rate()

    print(f'Epoch {epoch+1}/{num_epochs} finished. Time Taken: {time.time() - epoch_start_time} sec')
