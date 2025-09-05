import os, glob, random, cv2, torch, math
from tqdm import tqdm
import argparse
from src.models.get_model import get_model
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def reshape_transform(tensor):
    grid_size = int(math.sqrt(tensor.shape[1]))
    result = tensor.reshape(tensor.size(0), grid_size, grid_size, tensor.size(2))
    result = result.permute(0, 3, 1, 2)
    return result

def generate_xai_comparison(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.model_name}_best.pth")
    if not os.path.exists(checkpoint_path): return
    
    class_names = sorted(os.listdir(os.path.join(args.data_dir, 'test')))
    model = get_model(args.model_name, num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    if 'swin' in args.model_name:
        target_layers, reshape_function = [model.layers[-1].blocks[-1].norm2], reshape_transform
    else:
        target_layers, reshape_function = [model.conv_head], None
    
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers, reshape_transform=reshape_function)
    output_dir = os.path.join(args.output_dir, f"{args.model_name}_xai_comparison")
    os.makedirs(output_dir, exist_ok=True)

    for class_name in class_names:
        class_dir = os.path.join(args.data_dir, 'test', class_name)
        sample_paths = random.sample(glob.glob(os.path.join(class_dir, '*.png')), min(len(glob.glob(os.path.join(class_dir, '*.png'))), args.num_samples))
        for i, img_path in enumerate(tqdm(sample_paths, desc=f"Generating for '{class_name}'")):
            original_img_bgr = cv2.imread(img_path)
            resized_original_bgr = cv2.resize(original_img_bgr, (args.image_size, args.image_size))
            rgb_img = cv2.cvtColor(resized_original_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(output_dir, f"{class_name}_sample_{i+1}_original.png"), resized_original_bgr)
            cv2.imwrite(os.path.join(output_dir, f"{class_name}_sample_{i+1}_gradcam.png"), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate side-by-side original vs. Grad-CAM++ visualizations.")
    parser.add_argument('--model_name', type=str, required=True, choices=['efficientnetv2_s', 'swin_tiny', 'convnext_tiny'])
    parser.add_argument('--image_size', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--checkpoint_dir', type=str, default='outputs/checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs/xai_results')
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()
    generate_xai_comparison(args)