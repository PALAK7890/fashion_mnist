import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from model import MyNN
import numpy as np
import pandas as pd

st.set_page_config(
page_title="FashionAI",
page_icon="👕",
layout="wide"
)

st.markdown("""

<style>

.stApp{
background:#F7F9FC;
}

.block-container{
padding-top:2rem;
padding-left:6rem;
padding-right:6rem;
}

.result-card{
background:white;
padding:25px;
border-radius:25px;
box-shadow:0 4px 18px rgba(0,0,0,.06);
}

.hero{
background:white;
padding:30px;
border-radius:25px;
margin-bottom:25px;
box-shadow:0 4px 18px rgba(0,0,0,.06);
}

</style>

""", unsafe_allow_html=True)

st.markdown("""

<div class='hero'>
<h1>FashionAI</h1>
<p>CNN-based Fashion Classification using PyTorch</p>
</div>
""", unsafe_allow_html=True)

classes=[
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

device="cpu"

model=MyNN(1)

model.load_state_dict(
torch.load(
"fashion_cnn.pth",
map_location=device
)
)

model.eval()

transform=transforms.Compose([
transforms.Grayscale(),
transforms.Resize((28,28)),
transforms.ToTensor()
])

tab1,tab2=st.tabs(
["Upload","Camera"]
)

def predict(image):

```
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

return pred,probs
```

with tab1:

```
file=st.file_uploader(
"Upload Image",
type=["png","jpg","jpeg"]
)

if file:

    image=Image.open(file)

    pred,probs=predict(image)

    left,right=st.columns([1,1])

    with left:

        st.image(
        image,
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

        st.markdown(
        "</div>",
        unsafe_allow_html=True
        )

    st.subheader(
    "Probability Distribution"
    )

    df=pd.DataFrame(
    {
    "Class":classes,
    "Probability":probs.numpy()
    })

    st.bar_chart(
    df.set_index(
    "Class"
    )
    )
```

with tab2:

```
camera=st.camera_input(
"Capture Image"
)

if camera:

    image=Image.open(camera)

    pred,probs=predict(image)

    col1,col2=st.columns([1,1])

    with col1:

        st.image(
        image,
        use_container_width=True
        )

    with col2:

        st.metric(
        "Prediction",
        classes[pred]
        )

        st.metric(
        "Confidence",
        f"{probs[pred]*100:.2f}%"
        )
```
