import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import MyNN
import pandas as pd

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="FashionAI",
    page_icon="👕",
    layout="wide"
)

# ======================================================
# CSS
# ======================================================

st.markdown("""
<style>

.stApp{
    background:#F7F9FC;
}

.block-container{
    padding-top:2rem;
    padding-left:5rem;
    padding-right:5rem;
}

.hero{
    background:white;
    padding:30px;
    border-radius:25px;
    margin-bottom:30px;
    box-shadow:0px 4px 18px rgba(0,0,0,.06);
}

.hero h1{
    color:#1D1D1F;
    font-size:40px;
}

.hero p{
    color:#6E6E73;
}

.result-card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 18px rgba(0,0,0,.05);
}

[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 4px 18px rgba(0,0,0,.05);
}

[data-testid="stFileUploader"]{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 18px rgba(0,0,0,.04);
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# HERO
# ======================================================

st.markdown("""
<div class='hero'>

<h1>FashionAI</h1>

<p>
CNN-based Fashion Classification using PyTorch and Streamlit
</p>

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
# LOAD MODEL
# ======================================================

device = torch.device("cpu")

model = MyNN(1)

model.load_state_dict(
    torch.load(
        "fashion_cnn.pth",
        map_location=device
    )
)

model.eval()

# ======================================================
# TRANSFORM
# ======================================================

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# ======================================================
# PREDICTION FUNCTION
# ======================================================

def predict(image):

    img = transform(image)
    img = img.unsqueeze(0)

    with torch.no_grad():

        output = model(img)

        probs = torch.softmax(
            output,
            dim=1
        )[0]

    pred = torch.argmax(probs).item()

    return pred, probs


# ======================================================
# TABS
# ======================================================

tab1, tab2 = st.tabs([
    "Upload Image",
    "Camera"
])

# ======================================================
# UPLOAD TAB
# ======================================================

with tab1:

    uploaded = st.file_uploader(
        "Upload image",
        type=["png","jpg","jpeg"]
    )

    if uploaded:

        image = Image.open(uploaded)

        pred, probs = predict(image)

        left, right = st.columns([1,1])

        with left:

            st.image(
                image,
                caption="Input Image",
                use_container_width=True
            )

        with right:

            st.markdown(
                "<div class='result-card'>",
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction",
                classes[pred]
            )

            st.metric(
                "Confidence",
                f"{probs[pred]*100:.2f}%"
            )

            st.progress(
                float(probs[pred])
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

        st.subheader("Probability Distribution")

        df = pd.DataFrame({
            "Class": classes,
            "Probability": probs.numpy()
        })

        st.bar_chart(
            df.set_index("Class")
        )

# ======================================================
# CAMERA TAB
# ======================================================

with tab2:

    camera = st.camera_input(
        "Capture image"
    )

    if camera:

        image = Image.open(camera)

        pred, probs = predict(image)

        left, right = st.columns([1,1])

        with left:

            st.image(
                image,
                caption="Captured Image",
                use_container_width=True
            )

        with right:

            st.metric(
                "Prediction",
                classes[pred]
            )

            st.metric(
                "Confidence",
                f"{probs[pred]*100:.2f}%"
            )

            st.progress(
                float(probs[pred])
            )

        st.subheader("Probability Distribution")

        df = pd.DataFrame({
            "Class": classes,
            "Probability": probs.numpy()
        })

        st.bar_chart(
            df.set_index("Class")
        )