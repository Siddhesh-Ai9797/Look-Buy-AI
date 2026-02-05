import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


def pick_device():
    # MPS for Apple Silicon, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/blip_caption_one.py <image_path> <output_txt_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    if not img_path.exists():
        raise FileNotFoundError(img_path)

    device = pick_device()

    # BLIP caption model (free)
    model_id = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)

    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    # generation settings for stable captions
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=5,
            do_sample=False
        )

    caption = processor.decode(out[0], skip_special_tokens=True).strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(caption, encoding="utf-8")
    print(caption)


if __name__ == "__main__":
    main()
