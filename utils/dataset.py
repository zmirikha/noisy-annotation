from torch.utils.data import Dataset
import glob
import os
from PIL import Image

class Skin(Dataset):
    def __init__(self, img_root, gt_root, input_size=(96, 96), im_transform=None, target_transform=None):
        super().__init__()

        self.img_root = img_root
        self.gt_root = gt_root

        self.transform = im_transform
        self.target_transform = target_transform

        self.input_width = input_size[0]
        self.input_height = input_size[1]

        self.img_filenames = glob.glob(os.path.join(self.img_root, "*"))
        self.gt_filenames = glob.glob(os.path.join(self.gt_root, "*"))

    def __getitem__(self, index):

        im = Image.open(self.img_filenames[index]).convert('RGB')
        target = Image.open(self.gt_filenames[index]).convert('L')

        if self.transform is not None:
            im = self.transform(im)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return im, target

    def __len__(self):
        return len(self.img_filenames)




