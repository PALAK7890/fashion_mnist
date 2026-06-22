import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MyNN

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="FashionAI",
    page_icon="👕",
    layout="wide"
)

# ---------------- SIDEBAR ----------------

theme = st.sidebar.selectbox(
    "🎨 Theme",
    ["Dark Blue", "Light Blue"]
)

# ---------------- CUSTOM CSS ----------------

if theme == "Dark Blue":

    bg = "#081120"
    card = "rgba(255,255,255,0.06)"
    text = "white"
    accent = "#3b82f6"

else:

    bg = "#eaf4ff"
    card = "rgba(255,255,255,0.65)"
    text = "#0a2540"
    accent = "#4da3ff"


st.markdown(f"""
<style>

.stApp {{
background:{bg};
color:{text};
}}

.hero {{
padding:30px;
border-radius:25px;
background:linear-gradient(
135deg,
{accent},
#60a5fa
);
margin-bottom:30px;
color:white;
}}

.glass {{
padding:25px;
border-radius:20px;
background:{card};
backdrop-filter: blur(20px);
border:1px solid rgba(255,255,255,.1);
}}

.small-card {{
padding:15px;
border-radius:15px;
background:{card};
text-align:center;
margin-top:10px;
}}

</style>
""", unsafe_allow_html=True)


# ---------------- HERO ----------------

st.markdown("""
<div class='hero'>
<h1>👕 FashionAI Classifier</h1>

Real-time clothing classification using
CNN + PyTorch + Streamlit

</div>
""", unsafe_allow_html=True)

# ---------------- CLASSES ----------------

classes = [
"T-shirt/top",
"Trouser",
"Pullover",
"Dress",
"Coat",
"Sandal",
"Shirt",
"Sneaker",
"Bag",
"Ankle boot"
]

# ---------------- LOAD MODEL ----------------

device="cpu"

model=MyNN(1)

model.load_state_dict(
    torch.load(
        "fashion_cnn.pth",
        map_location=device
    )
)

model.eval()

# ---------------- TRANSFORM ----------------

transform=transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# ---------------- TABS ----------------

tab1,tab2,tab3=st.tabs(
[
"📤 Upload",
"📷 Camera",
"📘 About Model"
]
)

# ===================================================
# UPLOAD
# ===================================================

with tab1:

    uploaded=st.file_uploader(
        "Upload Image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image=Image.open(uploaded)

        col1,col2=st.columns([1,1])

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        img=transform(image)
        img=img.unsqueeze(0)

        with torch.no_grad():

            output=model(img)

            probs=torch.softmax(
                output,
                dim=1
            )[0]

        pred=torch.argmax(
            probs
        ).item()

        confidence=probs[pred].item()

        with col2:

            st.markdown(
            "<div class='glass'>",
            unsafe_allow_html=True
            )

            st.metric(
                "Prediction",
                classes[pred]
            )

            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

            st.progress(
                confidence
            )

            st.markdown(
            "</div>",
            unsafe_allow_html=True
            )

        st.subheader(
            "🏆 Top Predictions"
        )

        top3=torch.topk(
            probs,
            3
        )

        cols=st.columns(3)

        for idx,col in enumerate(cols):

            with col:

                class_idx=top3.indices[idx]

                score=top3.values[idx]

                st.markdown(
                f"""
                <div class='small-card'>
                <h4>{classes[class_idx]}</h4>
                <p>{score*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
                )

# ===================================================
# CAMERA
# ===================================================

with tab2:

    camera=st.camera_input(
        "Take Picture"
    )

    if camera:

        image=Image.open(
            camera
        )

        st.image(
            image,
            width=300
        )

        img=transform(
            image
        ).unsqueeze(0)

        with torch.no_grad():

            output=model(
                img
            )

            probs=torch.softmax(
                output,
                dim=1
            )[0]

        pred=torch.argmax(
            probs
        ).item()

        confidence=probs[
            pred
        ].item()

        st.success(
            f"Prediction: {classes[pred]}"
        )

        st.progress(
            confidence
        )

# ===================================================
# ABOUT
# ===================================================

with tab3:

    st.markdown("""
<div class='glass'>

### Architecture

CNN:
- Conv2D
- BatchNorm
- MaxPooling
- Fully Connected Layers
- Dropout

### Dataset

Fashion-MNIST

### Framework

PyTorch + Streamlit

</div>
""", unsafe_allow_html=True)