import torch
from code.utils.feature_extractor import FPN_FeatureExtractor

def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]


ckpt_dir = 'checkpoints/epoch=15-step=193199.ckpt'

def load_only_FPN_feature_extractor_from_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Create a new instance of the FPN_FeatureExtractor
    fpn = FPN_FeatureExtractor(out_ch=32)

    # Extract the state_dict of the FPN_FeatureExtractor from the checkpoint
    state_dict = checkpoint['state_dict']
    fpn_state_dict = {k.partition('feat_extractor.')[2]: v for k, v in state_dict.items() if k.startswith('feat_extractor.')}

    # Load the weights into the FPN_FeatureExtractor
    fpn.load_state_dict(fpn_state_dict)

    return fpn

feature_extraction_volrecon = load_only_FPN_feature_extractor_from_checkpoint(ckpt_dir)

class Pix2PixOptions:
    def __init__(self, feature_extraction_volrecon=None):
        self.isTrain = True
        self.name = "Pix2PixExperiment"
        self.gpu_ids = get_available_gpus()
        self.checkpoints_dir = "./checkpoints"
        self.model = "pix2pix"
        self.input_nc = 4
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
        self.load_size = 512
        self.crop_size = 512
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
        self.getIntermFeat = True
        
        self.no_lsgan = False
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        
        self.niter_fix_global = 20
        
        
        self.resize_or_crop = 'resize'
        self.loadSize = 512
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
        self.lambda_feat = 10.0

        self.gen_features = False


        ################### DPGAN params:
                # General options
        self.name = 'label2coco'
        self.seed = 43
        self.gpu_ids = get_available_gpus()
        self.checkpoints_dir = './checkpoints'
        self.no_spectral_norm = False
        self.batch_size = 1
        self.dataroot = './datasets/cityscapes/'
        self.dataset_mode = 'coco'
        self.no_flip = False

        # For generator
        self.num_res_blocks = 6
        self.channels_G = 32
        self.param_free_norm = 'syncbatch'
        self.spade_ks = 3
        self.no_EMA = False
        self.EMA_decay = 0.9999
        self.no_3dnoise = False
        self.z_dim = 32
        self.crop_size = 512
        self.aspect_ratio = 1.0
        self.semantic_nc = 4
        self.phase = 'train'
        self.label_nc = 3
        self.loaded_latest_iter = 1
        


        # For training
        self.freq_print = 1000
        self.freq_save_ckpt = 20000
        self.freq_save_latest = 10000
        self.freq_smooth_loss = 250
        self.freq_save_loss = 2500
        self.freq_fid = 5000
        self.continue_train = False
        self.which_iter = 'latest'
        self.num_epochs = 200
        self.beta1 = 0.0
        self.beta2 = 0.999
        self.lr_g = 0.0001
        self.lr_d = 0.0004
        self.channels_D = 32
        self.add_vgg_loss = False
        self.lambda_vgg = 10.0
        self.no_balancing_inloss = False
        self.no_labelmix = False
        self.lambda_labelmix = 10.0
        # For testing
        self.results_dir = './results/'
        self.ckpt_iter = 'best'
        self.feat_extr = load_only_FPN_feature_extractor_from_checkpoint(ckpt_dir)

