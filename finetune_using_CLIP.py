# Fine-tuning UNet with LoRA adapters and CLIP perceptual loss

import os
import torch
import time
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules import util as ldm_util
from torch.optim.lr_scheduler import ReduceLROnPlateau

# -------------------------------------------------------------------
# >>> add this near the very top of finetune_using_CLIP.py  <<<
# -------------------------------------------------------------------


def safe_checkpoint(func, inputs, params=None, flag=False):
    """Filter out None tensors so .detach() is never called on None."""
    # `inputs` is a tuple like (x, emb, maybe_more, maybe_None, …)
    inputs = tuple(x for x in inputs if x is not None)

    if flag:                     # when SD asks for no checkpoint
        return func(*inputs)

    # When `params` is a dict of kwargs for `func`
    if params is None:
        return torch.utils.checkpoint.checkpoint(func, *inputs)
    elif isinstance(params, dict):
        return torch.utils.checkpoint.checkpoint(func, *inputs, **params)
    else:                        # SD sometimes passes a single tensor instead of dict
        return torch.utils.checkpoint.checkpoint(func, *inputs, params)

ldm_util.checkpoint = safe_checkpoint

os.makedirs("logs", exist_ok=True)
# 1. Dataset for content-style pairs
class StyleContentDataset(Dataset):
    VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    def __init__(self, content_dir, style_dir, img_size=512):
        # only list real image files
        self.content_paths = sorted(
            f for f in os.listdir(content_dir)
            if f.lower().endswith(self.VALID_EXTS)
        )
        self.style_paths = sorted(
            f for f in os.listdir(style_dir)
            if f.lower().endswith(self.VALID_EXTS)
        )
        #
        self.transform = transforms.Compose([
            transforms.CenterCrop(min(img_size, img_size)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.content_dir = content_dir
        self.style_dir   = style_dir

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        c_path = os.path.join(self.content_dir, self.content_paths[idx])
        c = Image.open(c_path).convert("RGB")

        # choose a random style filename from the full style list
        style_filename = random.choice(self.style_paths)
        s_path = os.path.join(self.style_dir, style_filename)
        s = Image.open(s_path).convert("RGB")

        return self.transform(c), self.transform(s)
# 2. Load pretrained diffusion UNet
device = "cuda"
cfg  = OmegaConf.load("models/ldm/stable-diffusion-v1/v1-inference.yaml")
sd   = torch.load("models/ldm/stable-diffusion-v1/model.ckpt", map_location="cpu")["state_dict"]
model = instantiate_from_config(cfg.model).cuda().eval()
model.load_state_dict(sd, strict=False)
unet = model.model.diffusion_model

# 3. Apply LoRA to the UNet
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=[
        "attn2.to_q",  # only cross-attn Q
        "attn2.to_k",  # only cross-attn K
        "attn2.to_v",  # only cross-attn V
    ]
)
unet_lora = get_peft_model(unet, lora_config).to(device)
# Freeze all except LoRA adapters
for n, p in unet_lora.named_parameters():
    if "lora_" not in n:
        p.requires_grad = False

# 4. CLIP model for perceptual loss
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

def compute_clip_loss(gen_img, style_img):
    # expect gen_img/style_img in [0,1], shape [B,3,H,W]
    gen = F.interpolate(gen_img, size=(224,224), mode='bilinear', align_corners=False)
    sty = F.interpolate(style_img, size=(224,224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466,0.4578275,0.40821073], device=device).view(1,3,1,1)
    std  = torch.tensor([0.26862954,0.26130258,0.27577711], device=device).view(1,3,1,1)
    gen_norm = (gen - mean) / std
    sty_norm = (sty - mean) / std
    with torch.no_grad():
        gen_feat = clip_model.encode_image(gen_norm)
        sty_feat = clip_model.encode_image(sty_norm)
    return F.mse_loss(gen_feat, sty_feat)

# 5. Sampling utilities
diffusion = DDIMSampler(model)
diffusion.make_schedule(ddim_num_steps=50, ddim_eta=0.0)
betas = diffusion.betas       # shape [T]
alphas = 1 - betas               # shape [T]
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
ddim_ts_cuda = torch.from_numpy(diffusion.ddim_timesteps).to(device)

# 6. DataLoader
dataset = StyleContentDataset("downloads/data", "downloads/wikiart", img_size=512)
loader  = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 7. Optimizer
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet_lora.parameters()),
    lr=1e-4
)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
num_epochs  = 20
lambda_clip = 0.2

log_interval    = 50
noise_loss_list = []
clip_loss_list  = []
step_list       = []
global_step     = 0

# 8. Training loop
begin = time.time()

unet_lora.train()
for epoch in range(num_epochs):
    running_noise = 0.0
    running_clip = 0.0
    lr = optimizer.param_groups[0]["lr"]
    print(f" LR in this epoch = {lr:.2e}")
    for step, (content, style) in enumerate(loader, start=1):
        global_step += 1
        content, style = content.to(device), style.to(device)

        # 8.1 Encode to latent space
        with torch.no_grad():
            c_latent = model.get_first_stage_encoding(model.encode_first_stage(content))
            s_latent = model.get_first_stage_encoding(model.encode_first_stage(style))

        # 8.2 Add noise
        #t = torch.randint(0, diffusion.num_timesteps, (content.size(0),), device=device)
        #x_noisy, noise = diffusion.q_sample(c_latent, t)

        # sample an index into your 50‐step DDIM schedule
        n_ddim = ddim_ts_cuda.shape[0]  # e.g. 50
        idx = torch.randint(0, n_ddim, (content.size(0),), device=device)
        t = ddim_ts_cuda[idx]

        alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)

        noise = torch.randn_like(c_latent)  # ε ∼ N(0,I)
        x_noisy = torch.sqrt(alpha_t) * c_latent + \
                  torch.sqrt(1 - alpha_t) * noise

        #x_noisy, noise = model.model.diffusion_model.q_sample(c_latent, t)

        # 8.3 Predict noise with LoRA-augmented UNet
        B = content.size(0)
        uc = model.get_learned_conditioning([""] * B)  # B × 77 × 768
        pred_noise = unet_lora(x_noisy, t, context=uc)

        # 8.4 Noise prediction loss
        loss_noise = F.mse_loss(pred_noise, noise)
        x0_pred = (x_noisy - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()

        # 8.5 Reconstruct and decode to image for CLIP loss
        with torch.no_grad():

            img_recon = model.decode_first_stage(x0_pred)
            img_recon = torch.clamp((img_recon + 1.0) / 2.0, 0.0, 1.0)

        # 8.6 CLIP perceptual loss
        loss_clip = compute_clip_loss(img_recon, style)

        # 8.7 Backprop & optimize
        loss = loss_noise + lambda_clip * loss_clip
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_noise += loss_noise.item()
        running_clip += loss_clip.item()

        # Every `log_interval` steps, print & append to lists
        if step % log_interval == 0:
            avg_noise = running_noise / log_interval
            avg_clip = running_clip / log_interval
            step_list.append(global_step)
            noise_loss_list.append(avg_noise)
            clip_loss_list.append(avg_clip)

            print(f"Epoch {epoch + 1}/{num_epochs} | "
                  f"Step {step}/{len(loader)} (Global Step {global_step}) | "
                  f"Noise Loss: {avg_noise:.4f} | CLIP Loss: {avg_clip:.4f}")

            running_noise = 0.0
            running_clip = 0.0

    print(f"Epoch {epoch+1}/{num_epochs} — Loss: {loss.item():.4f}")
    epoch_clip = running_clip / len(loader)  # average over all batches
    scheduler.step(epoch_clip)

print('Total training time= ', time.time() - begin)

ckpt = {
    "epoch": num_epochs,
    "unet_lora_state_dict": unet_lora.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "clip_model_state_dict": clip_model.state_dict(),
    "lora_config": lora_config.to_dict(),
}
torch.save(ckpt, "finetuned_styleid_lora_clip.ckpt")
print("Saved checkpoint: finetuned_styleid_lora_clip.ckpt")

with open("logs/training_metrics.txt", "w") as f:
    f.write("step,noise_loss,clip_loss\n")
    for s, n, c in zip(step_list, noise_loss_list, clip_loss_list):
        f.write(f"{s},{n:.6f},{c:.6f}\n")

print("Saved checkpoint: params to plot")