import data.img_transforms as T
import data.spatial_transforms as ST
import data.temporal_transforms as TT
from torch.utils.data import DataLoader
from data.dataloader import DataLoaderX
from data.dataset_loader import ImageDataset, TrainImageDataset
from data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from data.datasets.ltcc import LTCC
from data.datasets.prcc import PRCC
from data.datasets.last import LaST
from data.datasets.ccvid import CCVID
from data.datasets.deepchange import DeepChange
from data.datasets.vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging


__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
    'vcclothes': VCClothes,
    'vcclothes_sc': VCClothesSameClothes,
    'vcclothes_cc': VCClothesClothesChanging,
    'last': LaST,
    'ccvid': CCVID,
    'deepchange': DeepChange,
}


def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, __factory.keys()))

    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = []
    transform_train_rgb = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_train_mask = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor()
    ])
    transform_train.append(transform_train_rgb)
    transform_train.append(transform_train_mask)
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    transform_train, transform_test = build_img_transforms(config)
    train_sampler = DistributedRandomIdentitySampler(dataset.train,
                                                     num_instances=config.DATA.NUM_INSTANCES,
                                                     seed=config.SEED)
    trainloader = DataLoaderX(dataset=TrainImageDataset(dataset.train, transform=transform_train),
                             sampler=train_sampler,
                             batch_size=config.DATA.TRAIN_BATCH, num_workers=config.DATA.NUM_WORKERS,
                             pin_memory=True, drop_last=True)

    galleryloader = DataLoaderX(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                               sampler=DistributedInferenceSampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=True, drop_last=False, shuffle=False)

    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoaderX(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query_same),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoaderX(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler
    else:
        queryloader = DataLoaderX(dataset=ImageDataset(dataset.query, transform=transform_test),
                                 sampler=DistributedInferenceSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        return trainloader, queryloader, galleryloader, dataset, train_sampler

    

    
