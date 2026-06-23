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
    padding:35px;
    border-radius:25px;
    margin-bottom:30px;
    box-shadow:0px 4px 18px rgba(0,0,0,.06);
}

.hero h1{
    color:#1D1D1F;
    font-size:42px;
}

.hero p{
    color:#6E6E73;
}

.card{
    background:white;
    padding:25px;
    border-radius:20px;
    box-shadow:0px 4px 18px rgba(0,0,0,.05);
}

.small-card{
    background:white;
    border-radius:15px;
    padding:15px;
    text-align:center;
    box-shadow:0px 4px 15px rgba(0,0,0,.04);
}

[data-testid="stFileUploader"]{
    background:white;
    padding:20px;
    border-radius:20px;
    box-shadow:0px 4px 15px rgba(0,0,0,.05);
}

[data-testid="metric-container"]{
    background:white;
    border-radius:15px;
    padding:10px;
    box-shadow:0px 4px 15px rgba(0,0,0,.04);
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
CNN-based Fashion Classification using PyTorch
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
# MODEL LOAD
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
# IMAGE TRANSFORM
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
# IMAGE UPLOAD
# ======================================================

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]
)

if uploaded:

    image = Image.open(uploaded)

    pred, probs = predict(image)

    left,right = st.columns([1,1])

    with left:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with right:

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
            f"{probs[pred]*100:.2f}%"
        )

        st.progress(
            float(probs[pred])
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.subheader("Top Predictions")

    top3 = torch.topk(probs,3)

    cols = st.columns(3)

    for idx,col in enumerate(cols):

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

    st.subheader("Probability Distribution")

    df = pd.DataFrame({
        "Class":classes,
        "Probability":probs.numpy()
    })

    st.bar_chart(
        df.set_index("Class")
    )