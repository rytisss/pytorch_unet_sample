import logging
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from unet import UNet


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.shape[0], full_img.shape[1]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # === Settings ===
    model_path = '/Users/rytis/Desktop/cvdlink/semi supervised/unet_sample/unet_carvana_scale1.0_epoch2.pth'
    input_image_path = 'input.jpg'
    output_mask_path = 'output_mask.png'
    out_threshold = 0.5
    bilinear = False
    n_classes = 2

    # === Load model ===
    net = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use CUDA if
    net.to(device=device)
    

    logging.info(f'Loading model from {model_path}')
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    dummy_input = torch.randn(1, 3, 512, 512)  # [B, C, H, W]
    output = net(dummy_input)
    print(f"Output shape: {output.shape}")
    # === Load image using OpenCV ===
    logging.info(f'Loading image from {input_image_path}')
    bgr_img = cv2.imread(input_image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # === Predict ===
    mask = predict_img(net=net,
                       full_img=rgb_img,
                       scale_factor=scale_factor,
                       out_threshold=out_threshold,
                       device=device)

    # === Save result ===
    result_img = mask_to_image(mask, mask_values)
    result_img.save(output_mask_path)
    logging.info(f'Mask saved to {output_mask_path}')

    # Optional: visualize
    # plot_img_and_mask(Image.fromarray(rgb_img), mask)
