import os
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.logging import get_logger
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from tqdm import tqdm


logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def pgd_attack(
    instance_prompt,
    models,
    tokenizer,
    noise_scheduler,
    vae,
    data_tensor: torch.Tensor,
    original_images: torch.Tensor,
    num_steps: int,
    pgd_alpha: float,
    pgd_eps: float,
):
    unet, text_encoder = models
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")

    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    perturbed_images = data_tensor.detach().clone()
    perturbed_images.requires_grad_(True)

    input_ids = tokenizer(
        instance_prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.repeat(len(data_tensor), 1)

    for step in tqdm(range(num_steps)):
        perturbed_images.requires_grad = True
        latents = vae.encode(
            perturbed_images.to(device, dtype=weight_dtype)
        ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor  # N=4, C, 64, 64

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = torch.randint(0, 200, (bsz,), device=latents.device)
        # timesteps = torch.randint(0, 100, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids.to(device))[0]

        # Predict the noise residual
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )

        # no prior preservation
        unet.zero_grad()
        text_encoder.zero_grad()
        # TODO: add prior loss here
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        # timesteps (4) -> (4, 1, 1, 1)
        # (4, C, H, W)
        # loss = torch.mean(torch.sqrt(timesteps.reshape(-1, 1, 1, 1)) * (model_pred.float() - target.float()) ** 2)
        loss.backward()

        alpha = pgd_alpha
        eps = pgd_eps

        adv_images = perturbed_images + alpha * perturbed_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=+eps)
        perturbed_images = torch.clamp(original_images + eta, min=-1, max=+1).detach_()
    return perturbed_images


def load_data(data_dir, size=512, center_crop=True) -> torch.Tensor:
    image_transforms = transforms.Compose(
        [
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [
        image_transforms(Image.open(i).convert("RGB"))
        for i in list(Path(data_dir).iterdir())
    ]
    images = torch.stack(images)
    return images


def init_model():
    PRETRAINED_MODELS_PATH = "/lustre/scratch/client/scratch/research/group/anhgroup/common/pretrained_models/stable-diffusion/stable-diffusion-2-1-base"

    text_encoder_cls = import_model_class_from_model_name_or_path(
        PRETRAINED_MODELS_PATH, None
    )

    # Load scheduler and models
    text_encoder = text_encoder_cls.from_pretrained(
        PRETRAINED_MODELS_PATH,
        subfolder="text_encoder",
        revision=None,
    )
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODELS_PATH, subfolder="unet", revision=None
    )
    unet = UNet2DConditionModel.from_pretrained(
        PRETRAINED_MODELS_PATH, subfolder="unet", revision=None
    )
    unet.enable_xformers_memory_efficient_attention()
    f = (unet, text_encoder)

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODELS_PATH,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        PRETRAINED_MODELS_PATH, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(
        PRETRAINED_MODELS_PATH, subfolder="vae", revision=None
    )

    return f, tokenizer, noise_scheduler, vae


def run(f, tokenizer, noise_scheduler, vae, original_data):
    pgd_alpha = 0.005
    pgd_eps = 0.05
    perturbed_data = original_data.clone()

    protected_images = pgd_attack(
        "a photo of sks person",
        f,
        tokenizer,
        noise_scheduler,
        vae,
        perturbed_data,
        original_data,
        num_steps=50,
        pgd_alpha=pgd_alpha,
        pgd_eps=pgd_eps,
    )

    noised_imgs = protected_images.detach()
    result = []
    for img in noised_imgs:
        result.append(
            Image.fromarray(
                (img * 127.5 + 128)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
        )
    return result


if __name__ == "__main__":
    IMAGES_TO_PROTECT_PATH = "/lustre/scratch/client/scratch/research/group/anhgroup/thanhlv19/EXPERIMENTS/DREAMBOOTH-FAST/code/test_images"
    OUTPUT_PATH = "./RESULT"
    f, tokenizer, noise_scheduler, vae = init_model()
    original_data = load_data(IMAGES_TO_PROTECT_PATH, size=512, center_crop=True)
    final_images = run(f, tokenizer, noise_scheduler, vae, original_data)
    save_folder = OUTPUT_PATH
    os.makedirs(save_folder, exist_ok=True)

    img_names = [
        str(instance_path).split("/")[-1]
        for instance_path in list(Path(IMAGES_TO_PROTECT_PATH).iterdir())
    ]
    for img, img_name in zip(final_images, img_names):
        save_path = os.path.join(save_folder, f"noise_{img_name}")
        img.save(save_path)
    print(f"Saved noise at {save_folder}")
