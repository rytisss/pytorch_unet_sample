import torch
import requests
import numpy as np
import cv2
import albumentations as A
import segmentation_models_pytorch as smp

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model and preprocessing function
    checkpoint = "smp-hub/segformer-b5-640x640-ade-160k"
    model = smp.from_pretrained(checkpoint).eval().to(device)
    preprocessing = A.Compose.from_pretrained(checkpoint)

    # Load image using OpenCV
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    response = requests.get(url, stream=True).raw
    img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Preprocess image
    normalized_image = preprocessing(image=image_rgb)["image"]
    input_tensor = torch.as_tensor(normalized_image).permute(2, 0, 1).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output_mask = model(input_tensor)

    # Postprocess mask
    mask = torch.nn.functional.interpolate(
        output_mask, size=(image_rgb.shape[0], image_rgb.shape[1]),
        mode="bilinear", align_corners=False
    )
    mask = mask.argmax(1).squeeze().cpu().numpy()

    # Convert back to BGR for visualization
    output_image = image_bgr.copy()

    # Extract contours for each class
    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # often background, optional to skip

        binary_mask = np.uint8(mask == class_id) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(output_image, contours, -1, color, thickness=2)

    # Show result
    cv2.imshow("Contours on Original Image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
