import argparse

import torch
from PIL import Image
from torchvision import transforms

from run_ablation import ResNet18WithEmbed

REGIONS = ["forehead", "chin", "nose", "left_cheek", "right_cheek"]
CLASS_NAMES = ["Mild", "Moderate", "Severe", "Very Severe"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights_path, device):
    model = ResNet18WithEmbed().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def predict_image(model, image_path, region, device):
    if region not in REGIONS:
        raise ValueError(f"Regiao invalida: {region}")

    image = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(image).unsqueeze(0).to(device)
    region_idx = torch.tensor([REGIONS.index(region)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tensor, region_idx)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())

    return {
        "region": region,
        "predicted_class": pred,
        "predicted_label": CLASS_NAMES[pred],
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }


def main():
    parser = argparse.ArgumentParser(description="Classificacao de gravidade da acne por regiao facial")
    parser.add_argument("image", help="Caminho da imagem de entrada")
    parser.add_argument("--region", required=True, choices=REGIONS, help="Regiao facial da imagem")
    parser.add_argument(
        "--weights",
        required=True,
        help="Caminho do arquivo .pth (ex: models/ablation_ckpt_clean_swa/with_embedding_seed42.pth)",
    )
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.weights, device)
    result = predict_image(model, args.image, args.region, device)

    print(f"Regiao: {result['region']}")
    print(f"Classe prevista: {result['predicted_label']} ({result['predicted_class']})")
    for label, prob in result["probabilities"].items():
        print(f"  {label}: {prob:.4f}")


if __name__ == "__main__":
    main()
