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

# --- Diffusers / Stable Diffusion Imports ---
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# --- Transformers / SegFormer Imports ---
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# --- Custom Dataset Import ---
from acdc_dataset import get_acdc_val_loaders

# --- Global Constants ---
NUM_CLASSES = 19
# ACDC Image Size
ACDC_IMG_SIZE = (1080, 1920)

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Standard Cityscapes Palette
CITYSCAPES_PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

# =================================================================================
# 1. Setup & Helpers (Seeding, Logging, Gaussian)
# =================================================================================

def setup_logging(log_dir="log"):
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"tta_steering_vae_seg_1024_512_w1_ss20_{current_time}.log")
    
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
    

def get_gaussian_kernel(patch_size, sigma_scale=1/8):
    h, w = patch_size
    y = torch.linspace(-(h - 1) / 2., (h - 1) / 2., h)
    sigma_y = sigma_scale * h
    gauss_y = torch.exp(-0.5 * (y / sigma_y) ** 2)
    
    x = torch.linspace(-(w - 1) / 2., (w - 1) / 2., w)
    sigma_x = sigma_scale * w
    gauss_x = torch.exp(-0.5 * (x / sigma_x) ** 2)
    
    kernel = torch.outer(gauss_y, gauss_x)
    kernel = kernel / kernel.max()
    return kernel

# =================================================================================
# 2. TTA / Steering Logic
# =================================================================================

def calculate_mean_entropy(logits):
    probs = F.softmax(logits, dim=1) + 1e-6
    log_probs = torch.log(probs)
    entropy_map = -torch.sum(probs * log_probs, dim=1) 
    return torch.mean(entropy_map)

def preprocess_for_segformer(image_tensor, processor, device):
    # image_tensor: [B, 3, H, W] in range [0, 1]
    # SegFormer expects specific mean/std normalization
    
    # Optional: Resize if patch size != 1024 (SegFormer native)
    # Since we are using 1024 patches, this is mostly a pass-through or safety
    if image_tensor.shape[-1] != 1024:
        image_tensor = F.interpolate(
            image_tensor, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        
    mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
    normalized_image = (image_tensor - mean) / std
    return normalized_image

@torch.no_grad()
def get_starting_latent(image_tensor, vae, scheduler, start_timestep, device, generator):
    # Encode
    dist = vae.encode(image_tensor.to(device).float()).latent_dist
    z0 = dist.sample(generator=generator)
    z0 = z0 * vae.config.scaling_factor
    
    batch_size = z0.shape[0]
    
    # Noise
    noise = torch.randn(z0.shape, device=device, generator=generator, dtype=z0.dtype)
    
    timesteps = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
    z_t = scheduler.add_noise(z0, noise, timesteps)
    
    return z_t

def adapt_image_with_steering(
    image, vae, unet, scheduler, segmenter, processor, uncond_embeds, args, device, generator
):
    guidance_scale = args.guidance_scale
    batch_size = image.shape[0]
    
    # --- FIX STARTS HERE ---
    
    # 1. Slice the timesteps!
    # We ignore the first 'start_timestep' steps.
    # Example: If steps=50 and start=20, we want the last 30 steps.
    original_timesteps = scheduler.timesteps
    timesteps = original_timesteps[args.start_timestep :]

    # 2. Get Initial Noisy Latent using the NEW first step
    with torch.no_grad():
        image_vae = (image * 2.0) - 1.0 
        
        # Now timesteps[0] is the noise level at step 20 (not step 0)
        start_timestep_t = timesteps[0] 
        
        latent = get_starting_latent(image_vae, vae, scheduler, start_timestep_t, device, generator)

    # 3. Denoising Loop (uses the sliced 'timesteps' list)
    for t in tqdm(timesteps, desc="Steering", leave=False):
        
        # Enable Gradients for Steering on the Latent itself
        latent = latent.detach().requires_grad_(True)
        
        timestep_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        # --- MODIFICATION START ---
        # A. Predict Noise (Detach from Graph)
        # We wrap this in no_grad. This prevents PyTorch from building the 
        # graph through the U-Net, saving memory and satisfying your requirement.
        with torch.no_grad():
            noise_pred = unet(
                latent, timestep_batch, encoder_hidden_states=uncond_embeds
            ).sample
        
        # B. Estimate x0 (Differentiable w.r.t Latent, constant w.r.t U-Net)
        alpha_prod_t = scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # The gradient will flow from x0_pred_latent -> latent directly,
        # treating noise_pred as a fixed scalar intercept.
        x0_pred_latent = (latent - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
        # --- MODIFICATION END ---
        
        # C. Decode to Image (Differentiable through VAE)
        x0_pred_image = vae.decode((x0_pred_latent / vae.config.scaling_factor).float()).sample
        x0_pred_image_01 = (x0_pred_image / 2.0) + 0.5
        
        # D. Get Segmentation Entropy (Differentiable through Segmenter)
        x0_pred_image_processed = preprocess_for_segformer(x0_pred_image_01, processor, device)
        logits = segmenter(x0_pred_image_processed).logits
        entropy = calculate_mean_entropy(logits.float())
        
        # 3. Calculate Gradient (Entropy -> Segmenter -> VAE -> Latent)
        grad = torch.autograd.grad(entropy, latent)[0]
        
        # Cleanup Graph
        del x0_pred_latent, x0_pred_image, x0_pred_image_processed, logits, entropy
        
        # 4. Update Latent (Steering Step)
        with torch.no_grad():
            # Subtract gradient (Minimize entropy)
            latent_steered = latent.detach() - (grad * guidance_scale)
            
            # Standard Scheduler Step 
            # We use the noise_pred calculated earlier.
            latent = scheduler.step(noise_pred, t, latent_steered).prev_sample

    # 5. Final Decode
    with torch.no_grad():
        final_image = vae.decode(latent / vae.config.scaling_factor).sample
        final_image = torch.clamp((final_image / 2.0) + 0.5, 0.0, 1.0)
        
    return final_image
# =================================================================================
# 3. Metrics & Visualization (Cityscapes Style)
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

def save_segmentation_vis(pred_tensor, label_tensor, diff_tensor, img_path, img_name, output_dir):
    """
    2x2 Grid: [Original | GT] / [Steered Diffusion | Pred]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    pred_mask = pred_tensor.squeeze().cpu().numpy().astype(np.uint8)
    label_mask = label_tensor.squeeze().cpu().numpy().astype(np.uint8)
    
    diff_img_np = diff_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    diff_img_np = (diff_img_np * 255).clip(0, 255).astype(np.uint8)
    
    h, w = pred_mask.shape

    # Load Original
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

    # Color Maps
    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    label_color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(len(CITYSCAPES_PALETTE)):
        pred_color[pred_mask == cls_id] = CITYSCAPES_PALETTE[cls_id]
        label_color[label_mask == cls_id] = CITYSCAPES_PALETTE[cls_id]

    # Overlays
    alpha = 0.4
    if original_img.max() == 0: 
        gt_overlay = label_color
        pred_overlay = pred_color
    else:
        gt_overlay = cv2.addWeighted(original_img, 1.0, label_color, alpha, 0)
        pred_overlay = cv2.addWeighted(original_img, 1.0, pred_color, alpha, 0)

    # Grid
    if diff_img_np.shape[:2] != (h, w):
         diff_img_np = cv2.resize(diff_img_np, (w, h))

    row1 = np.hstack((original_img, gt_overlay))
    row2 = np.hstack((diff_img_np, pred_overlay))
    combined = np.vstack((row1, row2))
    
    save_path = os.path.join(output_dir, f"{img_name}_vis.png")
    Image.fromarray(combined).save(save_path)

# =================================================================================
# 4. Main Logic
# =================================================================================

def create_argparser():
    defaults = dict(
        dataloader_batch_size=1, 
        max_images=10, 
        seed=42, 
        num_inference_steps=50, 
        start_timestep=20, # <--- Corresponds to 'start_step_idx' in previous code
        guidance_scale=1.0, # Entropy scale
        r="/datavolume/data/aysu/ACDC/rgb_anon", 
        g="/datavolume/data/aysu/ACDC/gt_trainval/gt", 
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        type_fn = type(v) if v is not None else int
        parser.add_argument(f"--{k}", default=v, type=type_fn)
    return parser

def main():
    args = create_argparser().parse_args()
    seed_everything(args.seed) 

    log_filename = setup_logging() 
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting TTA Steering Experiment with args: {args}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- Load Models ---
    sd_model_id = "runwayml/stable-diffusion-v1-5"
    logger.info(f"Loading SD Models...")
    vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet").to(device)
    scheduler = DDIMScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model_id, subfolder="text_encoder").to(device)
    
    # Configure Scheduler
    scheduler.set_timesteps(args.num_inference_steps)
    
    # Load SegFormer
    seg_model_id = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    logger.info(f"Loading SegFormer: {seg_model_id}")
    segmenter = SegformerForSemanticSegmentation.from_pretrained(seg_model_id).to(device).eval()
    processor = SegformerImageProcessor.from_pretrained(seg_model_id)

    # Enable Gradient Checkpointing (Crucial for TTA memory)
    unet.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    with torch.no_grad():
        tokens = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_embeds_base = text_encoder(tokens.input_ids.to(device))[0]

    # --- Load Data Config ---
    # OFFICIAL STITCHING CONFIG
    SD_PATCH_SIZE = (1024, 1024)   
    PATCH_STRIDE = (512, 512)    
    
    logger.info(f"Batch size is {SD_PATCH_SIZE}, patch stride is {PATCH_STRIDE}")
    patch_weight = get_gaussian_kernel(SD_PATCH_SIZE).to(device)

    val_loaders = get_acdc_val_loaders(
        base_image_dir=args.r, 
        base_mask_dir=args.g, 
        batch_size=args.dataloader_batch_size, 
        num_workers=4, 
        image_size=ACDC_IMG_SIZE,
        patch_size=SD_PATCH_SIZE,
        stride=PATCH_STRIDE,
        max_images=args.max_images,
        seed=args.seed 
    )

    for dataset_name, loader in val_loaders.items():
        logger.info(f"--- Processing {dataset_name} ---")
        
        all_stitched_preds = {}
        all_stitched_diff_imgs = {}
        all_stitched_labels = {}
        all_pred_counts = {}
        per_image_miou_list = []
        
        total_subset_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint64)
        
        current_processing_img_idx = -1
        last_processed_img_name = ""

        pbar = tqdm(loader, desc=f"TTA Steering {dataset_name}")
        
        for i, (image_patches, mask_patches, coords, img_names) in enumerate(pbar):
            
            image_patches = image_patches.to(device)
            current_batch_size = image_patches.shape[0]

            # Deterministic RNG for this batch
            batch_seed = args.seed + i
            batch_generator = torch.Generator(device=device).manual_seed(batch_seed)
            batch_emb = uncond_embeds_base.repeat(current_batch_size, 1, 1)
            
            # --- 1. RUN TTA STEERING ---
            # Returns the adapted (cleaned) image
            generated_patches = adapt_image_with_steering(
                image_patches, vae, unet, scheduler, segmenter, 
                processor, batch_emb, args, device, batch_generator
            )

            # --- 2. RUN INFERENCE ON ADAPTED IMAGE ---
            mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
            std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
            normalized_input = (generated_patches - mean) / std

            with torch.no_grad():
                outputs = segmenter(normalized_input)
                # Interpolate to patch size (1024)
                batch_logits = F.interpolate(
                    outputs.logits, size=SD_PATCH_SIZE, mode="bilinear", align_corners=False
                ).cpu()
                batch_labels = mask_patches.cpu()
                batch_diff_imgs = generated_patches.detach().cpu()

            # --- 3. STITCHING ---
            for b in range(current_batch_size):
                patch_img_idx = coords[0][b].item()
                y = coords[1][b].item()
                x = coords[2][b].item()
                patch_img_name = img_names[b]

                if current_processing_img_idx != -1 and patch_img_idx != current_processing_img_idx:
                    pred_logit_tensor = all_stitched_preds[current_processing_img_idx]
                    diff_img_tensor = all_stitched_diff_imgs[current_processing_img_idx]
                    label_tensor = all_stitched_labels[current_processing_img_idx]
                    count_tensor = all_pred_counts[current_processing_img_idx]
                    
                    counts_expanded = count_tensor.clamp(min=1e-6).unsqueeze(0)
                    
                    # Weighted Average
                    avg_logits = pred_logit_tensor / counts_expanded
                    final_pred = torch.argmax(avg_logits, dim=0).long().unsqueeze(0) 
                    final_label = label_tensor.unsqueeze(0) 
                    
                    final_diff_img = diff_img_tensor / counts_expanded 
                    final_diff_img = final_diff_img.unsqueeze(0) 

                    miou, cm, _ = compute_iou_and_cm(final_pred, final_label, NUM_CLASSES)
                    logger.info(f"Image {current_processing_img_idx} ({last_processed_img_name}) mIoU: {miou * 100:.2f}%")
                    
                    if len(per_image_miou_list) < 3:
                        seq_name = last_processed_img_name.split('_')[0]
                        full_img_path = os.path.join(args.r, dataset_name, "val", seq_name, last_processed_img_name)
                        save_segmentation_vis(
                            final_pred, final_label, final_diff_img, full_img_path,
                            f"{dataset_name}_{last_processed_img_name}",
                            output_dir="vis_results/tta_steering_vae_seg_1024_512_w1_ss20/"
                        )
                    
                    per_image_miou_list.append(miou)
                    total_subset_cm += cm 
                    
                    del all_stitched_preds[current_processing_img_idx]
                    del all_stitched_diff_imgs[current_processing_img_idx]
                    del all_stitched_labels[current_processing_img_idx]
                    del all_pred_counts[current_processing_img_idx]

                current_processing_img_idx = patch_img_idx
                last_processed_img_name = patch_img_name

                if patch_img_idx not in all_stitched_preds:
                    all_stitched_preds[patch_img_idx] = torch.zeros((NUM_CLASSES, *ACDC_IMG_SIZE), dtype=torch.float32)
                    all_stitched_diff_imgs[patch_img_idx] = torch.zeros((3, *ACDC_IMG_SIZE), dtype=torch.float32)
                    all_stitched_labels[patch_img_idx] = torch.full(ACDC_IMG_SIZE, 255, dtype=torch.int64)
                    all_pred_counts[patch_img_idx] = torch.zeros(ACDC_IMG_SIZE, dtype=torch.float32)

                patch_region_2d = (slice(y, y + SD_PATCH_SIZE[0]), slice(x, x + SD_PATCH_SIZE[1]))
                patch_region_3d = (slice(None), *patch_region_2d) 
                
                # Apply Gaussian Weights
                weight_cpu = patch_weight.cpu()
                weighted_logits = batch_logits[b] * weight_cpu
                weighted_diff = batch_diff_imgs[b] * weight_cpu
                
                all_stitched_preds[patch_img_idx][patch_region_3d] += weighted_logits
                all_stitched_diff_imgs[patch_img_idx][patch_region_3d] += weighted_diff
                all_pred_counts[patch_img_idx][patch_region_2d] += weight_cpu
                all_stitched_labels[patch_img_idx][patch_region_2d] = batch_labels[b]

        # --- Final Image ---
        if current_processing_img_idx != -1 and current_processing_img_idx in all_stitched_preds:
            pred_logit_tensor = all_stitched_preds[current_processing_img_idx]
            diff_img_tensor = all_stitched_diff_imgs[current_processing_img_idx]
            label_tensor = all_stitched_labels[current_processing_img_idx]
            count_tensor = all_pred_counts[current_processing_img_idx]
            
            counts_expanded = count_tensor.clamp(min=1e-6).unsqueeze(0)
            avg_logits = pred_logit_tensor / counts_expanded
            final_pred = torch.argmax(avg_logits, dim=0).long().unsqueeze(0)
            final_label = label_tensor.unsqueeze(0)

            final_diff_img = diff_img_tensor / counts_expanded
            final_diff_img = final_diff_img.unsqueeze(0)
            
            miou, cm, _ = compute_iou_and_cm(final_pred, final_label, NUM_CLASSES)
            logger.info(f"Image {current_processing_img_idx} ({last_processed_img_name}) mIoU: {miou * 100:.2f}%")
            
            if len(per_image_miou_list) < 3:
                seq_name = last_processed_img_name.split('_')[0]
                full_img_path = os.path.join(args.r, dataset_name, "val", seq_name, last_processed_img_name)
                save_segmentation_vis(final_pred, final_label, final_diff_img, full_img_path, f"{dataset_name}_{last_processed_img_name}", "vis_results/tta_steering_vae_seg_1024_512_w1_ss20/")

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
        logging.getLogger(__name__).error(f"Fatal error: {e}", exc_info=True)