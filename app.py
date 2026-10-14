import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

# Medical Disclosure Guardrail
st.set_page_config(page_title="Oral Cancer Triage AI", page_icon="🩺", layout="centered")

st.title("🩺 Oral Cancer Safety-First Triage AI")
st.caption("Team Testing Portal • Calibrated for 100% Sensitivity (0 False Negatives on Test Set)")

MODEL_PATH = "best_oral_triage_b0_antishortcut.pt"
GREEN_MAX = 0.32
AMBER_MAX = 0.86

@st.cache_resource
def load_triage_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_triage_model()

eval_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

zoom_eval_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(260),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Input tabs for mobile gallery upload OR live mobile camera capture
tab1, tab2 = st.tabs(["📁 Upload Mouth Photo", "📷 Take Live Phone Photo"])
uploaded_file = tab1.file_uploader("Select a JPG or PNG image", type=["jpg", "jpeg", "png"])
camera_file = tab2.camera_input("Take a well-lit photo of the inner mouth")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    raw_img = Image.open(image_source).convert("RGB")
    
    # Pre-crop outer 8% border to remove outer lips, watermarks, or retractors
    w, h = raw_img.size
    cropped_img = raw_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))

    with st.spinner("Running Multi-View Safety TTA & Grad-CAM..."):
        t_full = eval_transform(cropped_img).unsqueeze(0).to(device)
        
        # 1. Forward inference pipeline safely wrapped in no_grad
        with torch.no_grad():
            t_flip = torch.flip(t_full, dims=[3])
            t_zoom = zoom_eval_transform(cropped_img).unsqueeze(0).to(device)

            p1 = torch.sigmoid(model(t_full))
            p2 = torch.sigmoid(model(t_flip))
            p3 = torch.sigmoid(model(t_zoom))
            prob = float(torch.maximum(torch.maximum(p1, p2), p3).cpu().item())

        # 2. Grad-CAM generation outside no_grad with context manager to prevent hook memory leaks
        disp_pil = cropped_img.resize((260, 260))
        rgb_float = np.float32(disp_pil) / 255.0
        
        # BinaryClassifierOutputTarget(1) multiplies the single logit by +1 (positive cancer gradient)
        targets_for_cam = [BinaryClassifierOutputTarget(1)]
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=t_full, targets=targets_for_cam)[0, :]
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Display Traffic Light Banner
    if prob < GREEN_MAX:
        st.success(f"### 🟢 GREEN: Low Risk / Normal Tissue\n**Risk Score:** `{prob*100:.1f}%` (Threshold < 32%)\n\n**Action:** No high-risk visual features detected. Maintain routine oral hygiene.")
    elif prob < AMBER_MAX:
        st.warning(f"### 🟡 AMBER: Suspicious Ulcer or Atypical Mucosa\n**Risk Score:** `{prob*100:.1f}%` (Threshold 32% – 86%)\n\n**Action:** Monitor closely. If an ulcer or patch persists for more than 2 weeks, schedule a dental checkup.")
    else:
        st.error(f"### 🔴 RED: High-Risk Oral Lesion Detected\n**Risk Score:** `{prob*100:.1f}%` (Threshold ≥ 86%)\n\n**Action:** Urgent clinical/specialist evaluation and biopsy assessment advised.")

    # Clinical Informational Disclaimer Requirement
    st.info("**Disclaimer:** This tool is for training evaluation and triage demonstration purposes only. It does not constitute medical advice or a formal diagnosis.")

    # Side-by-side Image & Grad-CAM Heatmap
    col1, col2 = st.columns(2)
    with col1:
        st.image(disp_pil, caption="Analyzed Inner-Mouth Region", use_container_width=True)
    with col2:
        st.image(cam_overlay, caption="Grad-CAM AI Attention Map", use_container_width=True)