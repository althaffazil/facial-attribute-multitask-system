import torch
from torchvision import transforms

def load_model(model_class, model_path, device):
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image, device, image_size=128):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)[0].cpu().numpy()

    gender_prob = float(probs[0])
    smile_prob = float(probs[1])

    gender_label = "Male" if gender_prob >= 0.5 else "Female"
    smile_label = "Yes" if smile_prob >= 0.5 else "No"

    return {
        "gender_label": gender_label,
        "gender_conf": round(gender_prob * 100, 2),
        "smile_label": smile_label,
        "smile_conf": round(smile_prob * 100, 2),
    }
