import torch
from code.utils.feature_extractor import FPN_FeatureExtractor


def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]


ckpt_dir = 'checkpoints/epoch=15-step=193199.ckpt'


def load_only_FPN_feature_extractor_from_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create a new instance of the FPN_FeatureExtractor
    fpn = FPN_FeatureExtractor(out_ch=32)

    # Extract the state_dict of the FPN_FeatureExtractor from the checkpoint
    state_dict = checkpoint['state_dict']
    fpn_state_dict = {k.partition('feat_extractor.')[
        2]: v for k, v in state_dict.items() if k.startswith('feat_extractor.')}

    # Load the weights into the FPN_FeatureExtractor
    fpn.load_state_dict(fpn_state_dict)

    return fpn


class Pix2PixOptions:
    def __init__(self, feature_extraction_volrecon=None):
        self.isTrain = True #
        self.gpu_ids = get_available_gpus()
        self.input_nc = 4#
        self.output_nc = 3#
        self.ndf = 32#
        self.n_layers_D = 3 #
        self.netG = "local" #
        self.norm = "instance" #
        self.crop_size = 512 #
        self.no_flip = False #
        self.verbose = False #
        self.lr = 2e-4 #

        self.niter_decay = 100 #

        # multiscale_d opts
        self.use_sigmoid = False #
        self.num_D = 3 #
        self.getIntermFeat = True #

        self.n_downsample_global = 4 #
        self.n_local_enhancers = 1 #


        self.resize_or_crop = 'resize' #
        self.loadSize = 512 #
        self.fineSize = 512 #
        # Default arguments from TrainOptions
        self.lambda_L1 = 10.0#
        self.lambda_L2 = 0.0 # 
        self.lambda_VGG = 10.0 # 
        self.lambda_feat = 10.0# 


        # DPGAN params:
        # General options
        self.no_spectral_norm = False #
        self.no_flip = False # 

        # For generator
        self.num_res_blocks = 6 #
        self.channels_G = 32 #
        self.param_free_norm = 'syncbatch' #
        self.spade_ks = 3 #
        self.no_3dnoise = False #
        self.z_dim = 32 #
        self.crop_size = 512 #
        self.aspect_ratio = 1.0 #
        self.semantic_nc = 4 #

        # For training
        self.beta1 = 0.0#
        # For testing
        self.feat_extr = load_only_FPN_feature_extractor_from_checkpoint(
            ckpt_dir)#
