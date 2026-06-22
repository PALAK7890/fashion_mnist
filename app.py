import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MyNN

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="FashionAI",
    page_icon="👕",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

/* Main background */

.stApp{
    background-color:#F5F7FB;
}

/* Page spacing */

.block-container{
    padding-top:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* Hero section */

.hero{
    background:white;
    padding:35px;
    border-radius:28px;
    box-shadow:0px 6px 24px rgba(0,0,0,.05);
    margin-bottom:30px;
}

.hero-title{
    font-size:42px;
    font-weight:700;
    color:#1D1D1F;
}

.hero-sub{
    font-size:18px;
    color:#6E6E73;
}

/* Cards */

.card{
    background:white;
    border-radius:24px;
    padding:25px;
    box-shadow:0px 4px 20px rgba(0,0,0,.05);
}

/* Prediction mini cards */

.small-card{
    background:white;
    border-radius:18px;
    padding:15px;
    text-align:center;
    box-shadow:0px 3px 15px rgba(0,0,0,.04);
}

/* Upload section */

[data-testid="stFileUploader"]{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,.04);
}

/* Metrics */

[data-testid="metric-container"]{
    background:white;
    border-radius:20px;
    padding:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,.05);
}

/* Buttons */

.stButton button{
    background:#0071E3 !important;
    color:white !important;
    border-radius:14px !important;
    border:none !important;
    width:100%;
    height:50px;
    font-size:16px;
    font-weight:600;
}

.stButton button:hover{
    background:#0077ED !important;
}

/* Tabs */

button[data-baseweb="tab"]{
    font-size:16px;
    border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
👕 FashionAI
</div>

<div class="hero-sub">
Real-time fashion image classification using CNN + PyTorch
</div>

</div>
""", unsafe_allow_html=True)

# ======================================================
# CLASSES
# ======================================================

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

# ======================================================
# MODEL LOAD
# ======================================================

device = "cpu"

model = MyNN(1)

model.load_state_dict(
    torch.load(
        "fashion_cnn.pth",
        map_location=device
    )
)

model.eval()

# ======================================================
# IMAGE TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# ======================================================
# TABS
# ======================================================

tab1, tab2, tab3 = st.tabs([
    "📤 Upload",
    "📷 Camera",
    "📘 Model Info"
])

# ======================================================
# UPLOAD TAB
# ======================================================

with tab1:

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg","jpeg","png"]
    )

    if uploaded:

        image = Image.open(uploaded)

        col1, col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        img = transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            probs = torch.softmax(
                output,
                dim=1
            )[0]

        pred = torch.argmax(probs).item()
        confidence = probs[pred].item()

        with col2:

            st.markdown(
                "<div class='card'>",
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

            st.progress(confidence)

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.write("")
        st.subheader("Top Predictions")

        top3 = torch.topk(
            probs,
            3
        )

        cols = st.columns(3)

        for idx, col in enumerate(cols):

            with col:

                class_idx = top3.indices[idx]
                score = top3.values[idx]

                st.markdown(
                    f"""
                    <div class='small-card'>
                    <h4>{classes[class_idx]}</h4>
                    <p>{score*100:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ======================================================
# CAMERA TAB
# ======================================================

with tab2:

    camera = st.camera_input(
        "Take picture"
    )

    if camera:

        image = Image.open(camera)

        st.image(
            image,
            caption="Captured Image",
            width=350
        )

        img = transform(image)
        img = img.unsqueeze(0)

        with torch.no_grad():

            output = model(img)

            probs = torch.softmax(
                output,
                dim=1
            )[0]

        pred = torch.argmax(
            probs
        ).item()

        confidence = probs[
            pred
        ].item()

        st.metric(
            "Prediction",
            classes[pred]
        )

        st.progress(
            confidence
        )

# ======================================================
# MODEL INFO
# ======================================================

with tab3:

    st.markdown(
    """
<div class='card'>

### Model Architecture

CNN layers:

• Conv2D  
• ReLU  
• BatchNorm  
• MaxPooling  
• Fully Connected Layers  
• Dropout  

### Dataset

Fashion-MNIST

### Framework

PyTorch + Streamlit

### Notes

Designed for Fashion-MNIST images  
(28×28 grayscale images)

</div>
""",
unsafe_allow_html=True
)