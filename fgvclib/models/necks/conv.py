from ..backbones.inception import BasicConv2d
from fgvclib.models.necks import neck

@neck("conv")
def conv(cfg: dict):
    
    if cfg is not None:
        
        assert "num_features" in cfg.keys()
        assert isinstance(cfg["num_features"], int)
        assert "num_attentions" in cfg.keys()
        assert isinstance(cfg["num_attentions"], int) 
        assert "kernel_size" in cfg.keys()
        assert isinstance(cfg["kernel_size"], int) 

        return BasicConv2d(in_channels=cfg["num_features"], out_channels=cfg["num_attentions"], kernel_size=cfg["kernel_size"])
    
    return BasicConv2d()