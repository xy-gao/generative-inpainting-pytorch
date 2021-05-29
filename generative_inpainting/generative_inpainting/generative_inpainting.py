import os
import random
from argparse import ArgumentParser

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as vutils

from .model.networks import Generator
from .utils.tools import default_loader, get_model_list, normalize


class GenerativeInpainter:
    def __init__(
        self,
        checkpoint_path,
        netG_config={"input_dim": 3, "ngf": 32},
        cuda=False,
        device_ids=[],
        iter=0,
        seed=0,
    ):
        random.seed(seed)
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed_all(seed)
        self.cuda = cuda
        netG = Generator(netG_config, cuda, device_ids)
        # Resume weight
        last_model_name = get_model_list(checkpoint_path, "gen", iteration=iter)
        netG.load_state_dict(torch.load(last_model_name, map_location="cpu"))

        if cuda:
            netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
        self.netG = netG

    def _read(self, image_path, mask_path):
        self.x = default_loader(image_path)
        self.mask = default_loader(mask_path)

    def _proc(self, image_size):
        x = transforms.Resize(image_size)(self.x)
        x = transforms.CenterCrop(image_size)(x)
        mask = transforms.Resize(image_size)(self.mask)
        mask = transforms.CenterCrop(image_size)(mask)
        x = transforms.ToTensor()(x)
        mask = transforms.ToTensor()(mask)[0].unsqueeze(dim=0)
        x = normalize(x)
        x = x * (1.0 - mask)
        x = x.unsqueeze(dim=0)
        mask = mask.unsqueeze(dim=0)

        if self.cuda:
            x = x.cuda()
            mask = mask.cuda()
        x1, x2, offset_flow = self.netG(x, mask)
        self.inpainted_result = x2 * mask + x * (1.0 - mask)

    def _write(self, output_path):
        vutils.save_image(self.inpainted_result, output_path, padding=0, normalize=True)

    def __call__(self, image_path, mask_path, output_path, image_size=[512, 512]):
        self._read(image_path, mask_path)
        self._proc(image_size)
        self._write(output_path)
