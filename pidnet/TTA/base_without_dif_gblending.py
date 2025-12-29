import os
# This allows deterministic algorithms for CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import logging
import datetime
import random
import sys
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models

# --- Transformers / SegFormer Imports ---
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- Custom Dataset Import ---
from acdc_dataset import get_acdc_val_loaders

# --- Global Constants ---
NUM_CLASSES = 19
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Standard Cityscapes Palette for Visualization
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# =================================================================================
# 1. Helper Functions (Logging, Seeding, Gaussian)
# =================================================================================

def setup_logging(log_dir="log"):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"baseline_without_dif_gaussian_1024_512_{current_time}.log")
    
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return log_filename

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def get_gaussian_kernel(patch_size, sigma_scale=1/8):
    """
    Generates a 2D Gaussian Kernel for weighting patch predictions.
    
    """
    h, w = patch_size
    
    # Create 1D Gaussian for Y axis
    y = torch.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    sigma_y = sigma_scale * h
    gauss_y = torch.exp(-0.5 * (y / sigma_y) ** 2)
    
    # Create 1D Gaussian for X axis
    x = torch.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    sigma_x = sigma_scale * w
    gauss_x = torch.exp(-0.5 * (x / sigma_x) ** 2)
    
    # Create 2D Kernel (Outer Product)
    kernel = torch.outer(gauss_y, gauss_x)
    
    # Normalize (Max value = 1)
    kernel = kernel / kernel.max()
    
    return kernel

# =================================================================================
# 2. Visualization Helper (Overlay)
# =================================================================================
def save_segmentation_vis(pred_tensor, label_tensor, img_path, img_name, output_dir):
    """
    Saves a comparison: [Original Image | GT Overlay | Prediction Overlay]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Convert Tensors to Numpy
    pred_mask = pred_tensor.squeeze().cpu().numpy().astype(np.uint8)
    label_mask = label_tensor.squeeze().cpu().numpy().astype(np.uint8)
    h, w = pred_mask.shape

    # 2. Try to Load Original Image
    original_img = None
    if os.path.exists(img_path):
        try:
            original_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            if original_img.shape[:2] != (h, w):
                original_img = cv2.resize(original_img, (w, h))
        except Exception as e:
            print(f"Error loading image: {e}")
    else:
        print(f"DEBUG: Image not found at: {img_path}. Using black background.")

    if original_img is None:
        original_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 3. Create Color Maps
    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    label_color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(len(CITYSCAPES_PALETTE)):
        pred_color[pred_mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        label_color[label_mask == cls_id] = CITYSCAPES_PALETTE[cls_id]

    # 4. Create Overlays
    alpha = 0.4
    if original_img.max() == 0: 
        gt_overlay = label_color
        pred_overlay = pred_color
    else:
        gt_overlay = cv2.addWeighted(original_img, 1.0, label_color, alpha, 0)
        pred_overlay = cv2.addWeighted(original_img, 1.0, pred_color, alpha, 0)

    # 5. Stack
    combined = np.hstack((original_img, gt_overlay, pred_overlay))
    
    # 6. Save
    save_path = os.path.join(output_dir, f"{img_name}_vis.png")
    Image.fromarray(combined).save(save_path)


# =================================================================================
# 3. Evaluation Utilities (Cityscapes Logic)
# =================================================================================
def compute_iou_and_cm(all_preds, all_labels, num_classes, ignore_index=255):
    preds_flat = all_preds.view(-1).cpu().numpy()
    labels_flat = all_labels.view(-1).cpu().numpy()

    valid_indices = (labels_flat != ignore_index)
    preds_flat = preds_flat[valid_indices]
    labels_flat = labels_flat[valid_indices]

    cm = confusion_matrix(labels_flat, preds_flat, labels=list(range(num_classes))).astype(np.uint64)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    
    denominator = tp + fp + fn
    iou_per_class = np.full(num_classes, np.nan)
    valid_mask = denominator > 0
    iou_per_class[valid_mask] = tp[valid_mask] / denominator[valid_mask]

    miou = np.nanmean(iou_per_class)
    return miou, cm, iou_per_class


def calculate_metrics_from_total_cm(total_cm):
    tp = np.diag(total_cm)
    fp = total_cm.sum(axis=0) - tp
    fn = total_cm.sum(axis=1) - tp
    
    denominator = tp + fp + fn
    iou_per_class = np.full(len(tp), np.nan)
    valid_mask = denominator > 0
    iou_per_class[valid_mask] = tp[valid_mask] / denominator[valid_mask]
    
    global_miou = np.nanmean(iou_per_class)
    return global_miou, iou_per_class

# =================================================================================
# 4. Main Logic
# =================================================================================

def create_argparser():
    defaults = dict(
        dataloader_batch_size=1, 
        max_images=10, 
        seed=42, 
        r="/datavolume/data/aysu/ACDC/rgb_anon", 
        g="/datavolume/data/aysu/ACDC/gt_trainval/gt", 
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        type_fn = type(v) if v is not None else int
        parser.add_argument(f"--{k}", default=v, type=type_fn)
    return parser

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model

def main():
    args = create_argparser().parse_args()
    seed_everything(args.seed) 
    
    log_filename = setup_logging() 
    logger = logging.getLogger(__name__)
    logger.info(f"Starting BASELINE with Seed: {args.seed} | Max Images: {args.max_images}")
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # --- Load SegFormer ---
    num_classes = 19
    base_model = models.pidnet.get_pred_model('pidnet-l', num_classes=num_classes)
    model_path = "/home/moborobo/aysu/PIDNet/pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt"
    segmenter = load_pretrained(base_model, model_path).to(device).eval()

    logger.info(f"Loading Segmentation model: PIDNET-L")
    

    # --- Load Data Config ---
    ACDC_IMG_SIZE = (1080, 1920) 
    
    # 1. Set Patch Size to 1024 (Matches Training)
    SD_PATCH_SIZE = (1024, 1024)   
    
    # 2. Set Stride (512, 512) for 50% Overlap

    PATCH_STRIDE = (512, 512)   
    

    # 3. Generate Gaussian Kernel for Stitching
    logger.info(f"Generating Gaussian Kernel for patch size {SD_PATCH_SIZE} and stride {PATCH_STRIDE}")
    patch_weight = get_gaussian_kernel(SD_PATCH_SIZE).to(device)
    
    val_loaders = get_acdc_val_loaders(
        base_image_dir=args.r, 
        base_mask_dir=args.g, 
        batch_size=args.dataloader_batch_size, 
        num_workers=0, 
        image_size=ACDC_IMG_SIZE,
        patch_size=SD_PATCH_SIZE,
        stride=PATCH_STRIDE,
        max_images=args.max_images,
        seed=args.seed
    )

    # --- Processing Loop ---
    for dataset_name, loader in val_loaders.items():
        logger.info(f"--- Processing {dataset_name} ---")
        
        all_stitched_preds = {}
        all_stitched_labels = {}
        all_pred_counts = {}
        per_image_miou_list = []
        
        total_subset_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint64)
        
        current_processing_img_idx = -1
        last_processed_img_name = ""

        pbar = tqdm(loader, desc=f"Baseline {dataset_name}")
        
        for i, (image_patches, mask_patches, coords, img_names) in enumerate(pbar):
            
            patch_img_idx = coords[0][0].item()
            patch_img_name = img_names[0]

            # --- Evaluate Previous Image ---
            if current_processing_img_idx != -1 and patch_img_idx != current_processing_img_idx:
                pred_logit_tensor = all_stitched_preds[current_processing_img_idx]
                label_tensor = all_stitched_labels[current_processing_img_idx]
                count_tensor = all_pred_counts[current_processing_img_idx]
                
                counts_expanded = count_tensor.clamp(min=1e-6).unsqueeze(0)
                
                # Divide accumulated weighted logits by accumulated weights
                avg_logits = pred_logit_tensor / counts_expanded
                final_pred = torch.argmax(avg_logits, dim=0).long().unsqueeze(0) 
                final_label = label_tensor.unsqueeze(0) 

                miou, cm, _ = compute_iou_and_cm(final_pred, final_label, NUM_CLASSES)
                
                logger.info(f"Image {current_processing_img_idx} ({last_processed_img_name}) mIoU: {miou * 100:.2f}%")
                
                # --- VISUALIZATION ---
                if len(per_image_miou_list) < 3:
                     seq_name = last_processed_img_name.split('_')[0]
                     full_img_path = os.path.join(args.r, dataset_name, "val", seq_name, last_processed_img_name)
                     
                     save_segmentation_vis(
                        final_pred, 
                        final_label, 
                        full_img_path,
                        f"{dataset_name}_{last_processed_img_name}", 
                        output_dir="vis_results/base_without_dif_gaussian_1024/"
                     )
                # ---------------------

                per_image_miou_list.append(miou)
                total_subset_cm += cm
                
                del all_stitched_preds[current_processing_img_idx]
                del all_stitched_labels[current_processing_img_idx]
                del all_pred_counts[current_processing_img_idx]

            current_processing_img_idx = patch_img_idx
            last_processed_img_name = patch_img_name

            # --- Inference on Patch ---
            image_patches = image_patches.to(device) 

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            
            normalized_input = (image_patches - mean) / std

            with torch.no_grad():
                outputs = segmenter(normalized_input)
                # Ensure logits match patch size (1024x1024)
                # PIDNet returns the tensor directly, so we remove '.logits'
                patch_logits = F.interpolate(outputs, size=SD_PATCH_SIZE, mode="bilinear", align_corners=False).cpu()
                patch_labels = mask_patches.cpu()

            # --- Stitching with Gaussian Weights ---
            y = coords[1][0].item()
            x = coords[2][0].item()
            
            if patch_img_idx not in all_stitched_preds:
                all_stitched_preds[patch_img_idx] = torch.zeros((NUM_CLASSES, *ACDC_IMG_SIZE), dtype=torch.float32)
                all_stitched_labels[patch_img_idx] = torch.full(ACDC_IMG_SIZE, 255, dtype=torch.int64)
                all_pred_counts[patch_img_idx] = torch.zeros(ACDC_IMG_SIZE, dtype=torch.float32)

            patch_region_2d = (slice(y, y + SD_PATCH_SIZE[0]), slice(x, x + SD_PATCH_SIZE[1]))
            patch_region_3d = (slice(None), *patch_region_2d) 
            
            # Weighted Accumulation
            # Move weight to CPU to match storage tensors
            weight_cpu = patch_weight.cpu()
            
            # Multiply logits by the weight kernel
            weighted_logits = patch_logits[0] * weight_cpu
            
            all_stitched_preds[patch_img_idx][patch_region_3d] += weighted_logits 
            
            # Accumulate the WEIGHTS (not just +1.0)
            all_pred_counts[patch_img_idx][patch_region_2d] += weight_cpu
            
            all_stitched_labels[patch_img_idx][patch_region_2d] = patch_labels[0]

        # --- Process Final Image ---
        if current_processing_img_idx != -1 and current_processing_img_idx in all_stitched_preds:
            pred_logit_tensor = all_stitched_preds[current_processing_img_idx]
            label_tensor = all_stitched_labels[current_processing_img_idx]
            count_tensor = all_pred_counts[current_processing_img_idx]
            
            counts_expanded = count_tensor.clamp(min=1e-6).unsqueeze(0)
            avg_logits = pred_logit_tensor / counts_expanded
            final_pred = torch.argmax(avg_logits, dim=0).long().unsqueeze(0)
            final_label = label_tensor.unsqueeze(0)

            miou, cm, _ = compute_iou_and_cm(final_pred, final_label, NUM_CLASSES)
            logger.info(f"Image {current_processing_img_idx} ({last_processed_img_name}) mIoU: {miou * 100:.2f}%")
            
            if len(per_image_miou_list) < 3:
                seq_name = last_processed_img_name.split('_')[0]
                full_img_path = os.path.join(args.r, dataset_name, "val", seq_name, last_processed_img_name)
                save_segmentation_vis(
                    final_pred, 
                    final_label, 
                    full_img_path,
                    f"{dataset_name}_{last_processed_img_name}", 
                    "vis_results/base_without_dif_gaussian_1024/"
                )

            per_image_miou_list.append(miou)
            total_subset_cm += cm

        # --- Final Metrics ---
        logger.info(f"=== Results for {dataset_name} ===")
        if per_image_miou_list:
            avg_per_image_miou = np.mean(per_image_miou_list)
            logger.info(f"Average Per-Image mIoU: {avg_per_image_miou * 100:.2f}% (For Debugging)")
        
        global_miou, iou_per_class = calculate_metrics_from_total_cm(total_subset_cm)
        logger.info(f"Full Dataset mIoU:      {global_miou * 100:.2f}% (Official Cityscapes Metric)")
        
        logger.info("Per-Class IoU:")
        for cls_idx, iou in enumerate(iou_per_class):
             logger.info(f"  {CLASS_NAMES[cls_idx]:<15}: {iou * 100:.2f}%")

if __name__ == "__main__":
    seed_everything(42)
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")