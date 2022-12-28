from importlib import import_module

def import_backbone(cfg_backbone: dict):
    # used for backbone
    module_path = '.'.join(('src.models.backbone', cfg_backbone['file_name']))
    backbone_class = cfg_backbone['class_name']
    mod = import_module(module_path, package=backbone_class)
    mod = getattr(mod, backbone_class)
    return mod

def import_head(cfg_head: dict):
    # used for backbone
    module_path = 'src.models.head.elastic_face'
    head_class = cfg_head['class_name']
    mod = import_module(module_path, package=head_class)
    mod = getattr(mod, head_class)
    return mod

# from src.models.datamodule.[dataset___] import Siamese[Dataset__], Triplet[Dataset__]
def import_dataset(cfg: dict, is_train=True, custom_dataset_cls_prefix=None):
    module_path = '.'.join(('src.models.datamodule', 'dataset' + cfg['data']['dataset_postfix']))
    dataset_class = 'Dataset' + cfg['data']['dataset_postfix']

    if custom_dataset_cls_prefix:
        dataset_class = custom_dataset_cls_prefix + dataset_class
    mod = import_module(module_path, package=dataset_class)
    mod = getattr(mod, dataset_class)
    return mod


def import_network_pl(cfg, is_train=True):
    module_path = 'src.models.network_pl'
    mod = import_module(module_path, package='Network_pl')
    mod = getattr(mod, 'Network_pl')
    return mod