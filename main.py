import torch
from torch.backends import cudnn

from parameters import *
from trainer import Trainer
from tester import Tester
from data_loader import CustomDataLoader
from utils import make_folder
from augmentations import *


def main(config):
    cudnn.enabled = True
    # cudnn related setting
    # When the input data dimension or type of the network does not change much
    cudnn.benchmark = True
    cudnn.deterministic = False
    torch.cuda.manual_seed(2019)

    if config.train:
        # Create directories if not exist
        make_folder(config.model_save_path, config.arch)
        make_folder(config.sample_path, config.arch)

        transform = Compose([RandomHorizontallyFlip(p=.5), RandomSized(
            size=config.imsize), AdjustBrightness(bf=0.1), AdjustContrast(cf=0.1), AdjustHue(hue=0.1), AdjustSaturation(saturation=0.1)])
        data_loader = CustomDataLoader(config.img_path, config.label_path, config.imsize,
                                       config.batch_size, transform=transform, mode=config.train)
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()
    else:
        data_loader = CustomDataLoader(config.test_image_path, config.test_label_path, config.imsize,
                                       config.batch_size, mode=config.train)
        tester = Tester(data_loader.loader(), config)
        tester.test()


if __name__ == '__main__':
    config = get_parameters()
    main(config)
