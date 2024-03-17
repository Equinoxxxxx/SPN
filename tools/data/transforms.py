import random
import torch
import cv2
import torch.nn.functional as F
from torchvision.transforms import transforms
from torchvision.transforms import functional as tvf
from torchvision.transforms import InterpolationMode

class ComposeRandom(transforms.Compose):

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)
        self.flag = False
        self.randomize_parameters()

    def __call__(self, img):
        """
        Args:
            img : Image to be flipped.
        Returns:
            tensor: Randomly flipped image.
        """
        if self.random_p < self.p:
            self.flag = True
            return tvf.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = random.random()
        if self.random_p < self.p:
            self.flag = True
        else:
            self.flag = False


class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(self,
                 size,  # (h, w)
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),  # w / h
                 interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return tvf.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i, j, h, w)

    def randomize_parameters(self):
        self.randomize = True


def crop_local_ctx(ori_img, ltrb, tgt_size=(224, 224), interpo='bilinear'):
    '''
    ori_img: tensor: c, H, W
    '''
    _, H, W = ori_img.shape
    l, t, r, b = list(map(int, ltrb.detach()))
    x = (l+r) // 2
    y = (t+b) // 2
    h = b-t
    w = r-l
    crop_h = h*2
    crop_w = h*2
    crop_l = max(x-h, 0)
    crop_r = min(x+h, W)
    crop_t = max(y-h, 0)
    crop_b = min(y+h, W)
    cropped = ori_img[:, crop_t:crop_b, crop_l:crop_r]  # Chw
    l_pad = max(h-x, 0)
    r_pad = max(x+h-W, 0)
    t_pad = max(h-y, 0)
    b_pad = max(y+h-H, 0)
    cropped = F.pad(cropped, (l_pad, r_pad, t_pad, b_pad), 'constant', 0)
    assert cropped.size(1) == crop_h and cropped.size(2) == crop_w, (cropped.shape, (crop_h, crop_w))
    cropped = F.interpolate(torch.unsqueeze(cropped, dim=0).float(), size=tgt_size, mode=interpo)  # 1 C h w

    return torch.squeeze(cropped, dim=0).round()


def pad_keep_ratio(img, bbox, target_W=224, target_H=224, resize_mode='resize'):
    '''
    img: ndarray (h, w, C)
    bbox: list[l, t, r, b]
    '''
    l, t, r, b = map(int, bbox)
    h, w = img.shape[0], img.shape[1]
    if resize_mode == 'resize':
        resized = cv2.resize(img, (target_W, target_H))
    elif resize_mode == 'even_padded':
        if float(target_W) / target_H < float(w) / h:
            ratio = float(target_W) / w
        else:
            ratio = float(target_H) / h
        new_size = (int(w*ratio), int(h*ratio))
        cropped = cv2.resize(img, new_size)
        w_pad = target_W - new_size[0]
        h_pad = target_H - new_size[1]
        resized = cv2.copyMakeBorder(cropped,
                                    0,h_pad,0,w_pad,
                                    cv2.BORDER_CONSTANT,value=(0, 0, 0))  # t, b, l, r
    else:
        raise ValueError(resize_mode)
    return resized  # H, W, C


if __name__ == '__main__':
    t = RandomHorizontalFlip()
    imgs = torch.zeros([1, 1, 2, 2, 1])
    
    imgs[0, 0, 0, 0, 0] = 1
    print(imgs)

    print(t.flag)
    print(t(imgs.permute(4, 0, 1, 2, 3)).permute(1, 2, 3, 4, 0))