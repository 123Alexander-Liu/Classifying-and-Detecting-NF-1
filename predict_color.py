
import argparse
import logging
import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from utils.colors import get_colors
from config import UNetConfig

cfg = UNetConfig()

def inference_one(net, image, device):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img) 
        if cfg.deepsupervision: #采用深度监督学习，则只取最后一张图
            #output = output[-1]
            output = output[0][-1]

      
        probs = F.softmax(output, dim=1) #softmax分类


        probs = probs.squeeze(0)        # C x H x W,C个通道对应C个分类

        tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((image.size[1], image.size[0])),
                    transforms.ToTensor()
                ]
        )


        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask > cfg.out_threshold #超过阈值则为1，其余为0
            masks.append(mask)
        return masks  #输出C张图，表示图像各个像素点的6个归属类别


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='data/checkpoints/epoch_199.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', dest='input', type=str, default='data/test/images',
                        help='Directory of input images')
    parser.add_argument('--output', '-o', dest='output', type=str, default='data/test/outputs',
                        help='Directory of ouput images')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args) 
    input_imgs = os.listdir(args.input) #获取待测图像文件夹下所有文件名

    net = eval(cfg.model)(cfg)  #初始化NestedUNet类
    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)) #加载训练好的模型文件

    logging.info("Model loaded !")

    for i, img_name in tqdm(enumerate(input_imgs)):
        logging.info("\nPredicting image {} ...".format(img_name))

        img_path = osp.join(args.input, img_name)
        print(img_name)
        img = Image.open(img_path) #读取图像

        mask = inference_one(net=net,
                             image=img,
                             device=device) #前向推导，
        img_name_no_ext = osp.splitext(img_name)[0]
        output_img_dir = osp.join(args.output, img_name_no_ext)
        os.makedirs(output_img_dir, exist_ok=True)


        colors = get_colors(n_classes=cfg.n_classes) #获取表示像素C个类别对应的颜色RGB值
        w, h = img.size
        img_mask = np.zeros([h, w, 3], np.uint8)
        for idx in range(0, len(mask)): #遍历C张图，对于为（x,y）初的像素值，值为1则表示输出图像对应像素点为该类别
            image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8)) # 像素值1则变为255
            array_img = np.asarray(image_idx)
            img_mask[np.where(array_img==255)] = colors[idx] 
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)
        output = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
        #cv2.imwrite(osp.join(output_img_dir, img_name), output)
        cv2.imwrite(osp.join(output_img_dir, img_name), img_mask) #保存掩模图像，表示输出结果

