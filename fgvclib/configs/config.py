from yacs.config import CfgNode as CN


class FGVCConfig(object):

    def __init__(self):
        
        self.cfg = CN()

        # Name of experiment
        self.cfg.EXP_NAME = None

        # Resume last train
        self.cfg.RESUME_WEIGHT = None

        # Use cuda
        self.cfg.USE_CUDA = True

        self.cfg.LOGGER = CN()
        self.cfg.LOGGER.NAME = "WandbLogger"
        self.cfg.LOGGER.FILE_PATH = "./logs/"

        # Datasets and data loader
        self.cfg.DATASETS = CN()
        self.cfg.DATASETS.ROOT = None 
        self.cfg.DATASETS.TRAIN = CN()
        self.cfg.DATASETS.TEST = CN()

        # train dataset and data loder
        self.cfg.DATASETS.TRAIN.BATCH_SIZE = 32
        self.cfg.DATASETS.TRAIN.POSITIVE = None
        self.cfg.DATASETS.TRAIN.PIN_MEMORY = True
        self.cfg.DATASETS.TRAIN.SHUFFLE = True
        self.cfg.DATASETS.TRAIN.NUM_WORKERS = 0
        
        # test dataset and data loder
        self.cfg.DATASETS.TEST.BATCH_SIZE = 32
        self.cfg.DATASETS.TEST.POSITIVE = None
        self.cfg.DATASETS.TEST.PIN_MEMORY = False
        self.cfg.DATASETS.TEST.SHUFFLE = False
        self.cfg.DATASETS.TEST.NUM_WORKERS = 0

        # Model architecture
        self.cfg.MODEL = CN()
        self.cfg.MODEL.NAME = None
        self.cfg.MODEL.CLASS_NUM = None
        self.cfg.MODEL.OUTPUTS_NUM = None
        self.cfg.MODEL.CRITERIONS = None

        # Standard modulars of each model
        self.cfg.MODEL.BACKBONE = CN()
        self.cfg.MODEL.ENCODING = CN()
        self.cfg.MODEL.NECKS = CN()
        self.cfg.MODEL.HEADS = CN()
        
        # Setting of backbone
        self.cfg.MODEL.BACKBONE.NAME = None
        self.cfg.MODEL.BACKBONE.PRETRAINED = True
        self.cfg.MODEL.BACKBONE.ARGS = None

        # Setting of encoding
        self.cfg.MODEL.ENCODING.NAME = None
        self.cfg.MODEL.ENCODING.ARGS = None

        # Setting of neck
        self.cfg.MODEL.NECKS.NAME = None
        self.cfg.MODEL.NECKS.ARGS = None

        # Setting of head
        self.cfg.MODEL.HEADS.NAME = None
        self.cfg.MODEL.HEADS.ARGS = None
        
        # Transforms
        self.cfg.TRANSFORMS = CN()
        self.cfg.TRANSFORMS.TRAIN = None
        self.cfg.TRANSFORMS.TEST = None

        # Optimizer
        self.cfg.OPTIMIZER = CN()
        self.cfg.OPTIMIZER.NAME = "SGD"
        self.cfg.OPTIMIZER.MOMENTUM = 0.9
        self.cfg.OPTIMIZER.WEIGHT_DECAY = 5e-4
        self.cfg.OPTIMIZER.LR = CN()
        self.cfg.OPTIMIZER.LR.backbone = None
        self.cfg.OPTIMIZER.LR.encoding = None
        self.cfg.OPTIMIZER.LR.necks = None
        self.cfg.OPTIMIZER.LR.heads = None

        # Train
        self.cfg.ITERATION_NUM = None
        self.cfg.EPOCH_NUM = None
        self.cfg.START_EPOCH = None
        self.cfg.UPDATE_STRATEGY = None
        
        
        # Validation
        self.cfg.PER_ITERATION = None
        self.cfg.PER_EPOCH = None
        self.cfg.METRICS = None
    
    def get_cfg(self):
        return  self.cfg.clone()

    def load(self,config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()
    
    def stringfy():
        return 
    