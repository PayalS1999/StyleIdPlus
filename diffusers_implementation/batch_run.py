import os
import subprocess
import matplotlib.pyplot as plt
from PIL import Image

# Define lists of files and gamma values.
# (Assuming your style images are in ../data/sty and content images in ../data/cnt relative to current directory)
style_images = ["03.png", "10.png", "13.png", "21.png", "25.png", "28.png", "37.png"]
content_images = ["00.png", "06.png", "19.png"]
gamma_values = [0.2, 0.3]

# Base directories (adjust if necessary)
base_style_dir = os.path.join("..", "data", "sty")
base_content_dir = os.path.join("..", "data", "cnt")
# Output directory for generated images:
output_dir = os.path.join("..", "output", "batch_run")
os.makedirs(output_dir, exist_ok=True)

# We will store the generated image filenames in a dictionary keyed by gamma value.
results = {gamma: [] for gamma in gamma_values}

# Loop over content images, style images, and gamma values.
# (If you really need 21 images total, adjust these loops accordingly.)
for cnt in content_images:
    for sty in style_images:
        for gamma in gamma_values:
            cnt_path = os.path.join(base_content_dir, cnt)
            sty_path = os.path.join(base_style_dir, sty)
            # Create an output filename that encodes the input names and gamma value.
            output_fname = f"stylized_cnt{cnt.split('.')[0]}_sty{sty.split('.')[0]}_gamma{gamma}.png"
            # Here we assume run_styleid_diffusers.py uses the --save_dir flag to save outputs.
            # The script will likely save a final file named "stylized_image.jpg" in that folder;
            # For this batch, we rename the output after generation.
            temp_save_dir = output_dir  # using same folder for all outputs
            
            # Build the command to call run_styleid_diffusers.py:
            cmd = [
                "python", "run_styleid_diffusers.py",
                "--cnt_fn", cnt_path,
                "--sty_fn", sty_path,
                "--save_dir", temp_save_dir,
                "--ddim_steps", "50",
                "--gamma", str(gamma),
                "--T", "1.5"
            ]
            print("Running:", " ".join(cmd))
            # Run the command (this will generate the stylized image).
            subprocess.run(cmd, check=True)
            
            # Assume the script saves the final output as "stylized_image.jpg" in the save_dir.
            # Rename/move it to our desired name.
            src_path = os.path.join(temp_save_dir, "stylized_image.jpg")
            dst_path = os.path.join(temp_save_dir, output_fname)
            if os.path.exists(src_path):
                os.rename(src_path, dst_path)
                results[gamma].append(dst_path)
            else:
                print(f"Error: {src_path} not found.")

# --- Plotting the results ---
# For each gamma, create a grid where rows correspond to content images
# and columns correspond to style images.
for gamma in gamma_values:
    # Create a figure with rows = number of content images, columns = number of style images.
    n_rows = len(content_images)
    n_cols = len(style_images)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    # Filter results for this gamma.
    # We assume that for each content image, there is one image per style image in order.
    # Sort or arrange appropriately.
    image_list = results[gamma]
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx < len(image_list):
                img_path = image_list[idx]
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axs[i, j].imshow(img)
                    axs[i, j].axis("off")
                    axs[i, j].set_title(f"Content: {content_images[i]}, Style: {style_images[j]}")
                else:
                    axs[i, j].text(0.5, 0.5, "Missing", horizontalalignment='center', verticalalignment='center')
            idx += 1
    plt.suptitle(f"Results for gamma = {gamma}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"grid_gamma_{gamma}.png"))
    plt.show()
