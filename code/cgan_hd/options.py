import torch

def get_available_gpus():
    return [i for i in range(torch.cuda.device_count())]

class TrainOptions:
    def __init__(self):
        # Basic parameters
        self.name = 'label2city'
        self.gpu_ids = get_available_gpus()
        self.checkpoints_dir = './checkpoints'
        self.model = 'pix2pixHD'
        self.norm = 'instance'
        self.use_dropout = False
        self.data_type = 32
        self.verbose = False
        self.fp16 = False
        self.local_rank = 0

        # Input/output sizes
        self.batchSize = 1
        self.loadSize = 1024
        self.fineSize = 512
        self.label_nc = 35
        self.input_nc = 3
        self.output_nc = 3

        # For setting inputs
        self.dataroot = './datasets/cityscapes/'
        self.resize_or_crop = 'scale_width'
        self.serial_batches = False
        self.no_flip = False
        self.nThreads = 2
        self.max_dataset_size = float("inf")

        # For displays
        self.display_winsize = 512
        self.tf_log = False

        # For generator
        self.netG = 'global'
        self.ngf = 64
        self.n_downsample_global = 4
        self.n_blocks_global = 9
        self.n_blocks_local = 3
        self.n_local_enhancers = 1
        self.niter_fix_global = 0

        # For instance-wise features
        self.no_instance = False
        self.instance_feat = False
        self.label_feat = False
        self.feat_num = 3
        self.load_features = False
        self.n_downsample_E = 4
        self.nef = 16
        self.n_clusters = 10

        # Frequency settings
        self.display_freq = 100
        self.print_freq = 100
        self.save_latest_freq = 1000
        self.save_epoch_freq = 10
        self.no_html = False
        self.debug = False

        # For training
        self.continue_train = False
        self.load_pretrain = ''
        self.which_epoch = 'latest'
        self.phase = 'train'
        self.niter = 100
        self.niter_decay = 100
        self.beta1 = 0.5
        self.lr = 0.0002

        # For discriminators
        self.num_D = 2
        self.n_layers_D = 3
        self.ndf = 64
        self.lambda_feat = 10.0
        self.no_ganFeat_loss = False
        self.no_vgg_loss = False
        self.no_lsgan = False
        self.pool_size = 0
