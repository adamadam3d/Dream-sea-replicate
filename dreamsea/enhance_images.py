import cv2
import numpy as np
import argparse
from pathlib import Path

def gray_world_white_balance(img):
    """
    Applies Gray World White Balancing to an image.
    Assumption: The average color of the scene should be gray.
    This neutralizes color casts (e.g., the strong blue/green tint in underwater images)
    and inherently adjusts the perceived exposure.
    """
    # Convert to float32 to prevent overflow during calculations
    result = img.astype(np.float32)
    
    # Calculate the average values for each channel (B, G, R)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    # Calculate the global average
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    # Scale each channel by the ratio of the global average to its channel average.
    # We add a tiny epsilon (1e-6) to avoid division by zero.
    result[:, :, 0] = result[:, :, 0] * (avg_gray / (avg_b + 1e-6))
    result[:, :, 1] = result[:, :, 1] * (avg_gray / (avg_g + 1e-6))
    result[:, :, 2] = result[:, :, 2] * (avg_gray / (avg_r + 1e-6))
    
    # Fix Overexposure: If the scaling pushed values above 255, 
    # scale the entire image down so the brightest pixel hits exactly 255.
    # This preserves the color ratios without clipping (blowing out) the highlights.
    max_val = np.max(result)
    if max_val > 255.0:
        result = result * (255.0 / max_val)
    
    # Convert back to uint8
    return result.astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Converts to LAB color space, applies CLAHE to the Lightness channel,
    and converts back to BGR. This enhances local contrast and details 
    without drastically shifting the colors.
    """
    # Convert from BGR to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    # Apply CLAHE to the L (Lightness) channel
    cl = clahe.apply(l)
    
    # Merge the enhanced L channel with the original A and B channels
    limg = cv2.merge((cl, a, b))
    
    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def apply_msrcr(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0):
    """
    Applies Multi-Scale Retinex with Color Restoration (MSRCR).
    Mimics human color constancy. Excellent for removing varying illumination
    like AUV active strobe lights hotspots.
    """
    # 1. Convert to float and add 1 to avoid log(0)
    img_float = img.astype(np.float32) + 1.0
    
    # 2. Extract log-transformed channels and apply multi-scale retinex (MSR)
    msr = np.zeros_like(img_float)
    for sigma in sigmas:
        # Gaussian blur to estimate illumination
        blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
        # Subtract log-blurred from log-original
        msr += np.log10(img_float) - np.log10(blur)
    
    # Average the scales
    msr = msr / len(sigmas)
    
    # 3. Compute Color Restoration Function (CRF) to prevent desaturation
    img_sum = np.sum(img_float, axis=2, keepdims=True)
    crf = beta * (np.log10(alpha * img_float) - np.log10(img_sum))
    
    # 4. Multiply MSR by CRF
    msrcr = msr * crf
    
    # 5. Final linear contrast stretch to [0, 255]
    result = np.zeros_like(img_float)
    for i in range(3):
        channel = msrcr[:, :, i]
        # Use 1st and 99th percentiles to avoid outliers ruining the stretch
        min_val = np.percentile(channel, 1)
        max_val = np.percentile(channel, 99)
        channel_stretched = np.clip((channel - min_val) / (max_val - min_val + 1e-6) * 255.0, 0, 255)
        result[:, :, i] = channel_stretched
        
    return result.astype(np.uint8)

def process_image(img_path, output_path, use_gw=True, use_clahe=True, use_msrcr=False):
    """Reads an image, applies selected enhancements, and saves it."""
    # Read image in BGR format
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error loading image: {img_path}")
        return False
        
    result = img.copy()
    
    if use_msrcr:
        # MSRCR is a comprehensive enhancement, usually applied on its own.
        result = apply_msrcr(result)
    else:
        # 1. Neutralize color casts (Gray World)
        if use_gw:
            result = gray_world_white_balance(result)
            
        # 2. Enhance local contrast (CLAHE)
        if use_clahe:
            result = apply_clahe(result)
        
    # Save output
    cv2.imwrite(str(output_path), result)
    return True

def main():
    parser = argparse.ArgumentParser(description="Underwater Image Enhancement (Gray World + CLAHE or MSRCR)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save enhanced image(s)")
    parser.add_argument("--no_gw", action="store_true", help="Disable Gray World White Balancing")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE enhancement")
    parser.add_argument("--use_msrcr", action="store_true", help="Use Multi-Scale Retinex with Color Restoration instead of GW+CLAHE")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    use_gw = not args.no_gw
    use_clahe = not args.no_clahe
    use_msrcr = args.use_msrcr
    
    if input_path.is_file():
        # Process a single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If output is an existing directory, save inside it with original name
        if output_path.is_dir():
            output_path = output_path / input_path.name
            
        success = process_image(input_path, output_path, use_gw, use_clahe, use_msrcr)
        if success:
            print(f"Successfully enhanced: {output_path}")
            
    elif input_path.is_dir():
        # Process a directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_path.glob(ext))
            image_paths.extend(input_path.glob(ext.upper()))
            
        if not image_paths:
            print(f"No images found in {input_path}")
            return
            
        print(f"Found {len(image_paths)} images. Processing...")
        
        success_count = 0
        for p in image_paths:
            out_file = output_path / p.name
            if process_image(p, out_file, use_gw, use_clahe, use_msrcr):
                success_count += 1
                
        print(f"\nFinished processing! Successfully enhanced {success_count}/{len(image_paths)} images.")
        print(f"Output saved to: {output_path}")
    else:
        print(f"Input path does not exist: {input_path}")

if __name__ == "__main__":
    main()
