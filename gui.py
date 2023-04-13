import streamlit as st
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import base64
from torchvision import transforms
from streamlit_modal import Modal
import streamlit.components.v1 as components
from st_clickable_images import clickable_images

if "image_buffers" not in st.session_state:
    st.session_state.image_buffers = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SIZE = 512

image_transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

st.set_page_config(layout="wide", page_title="Gennaissance")
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: grey;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Gennaissance. We will never store your data on our server. Your privacy matters.</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

st.write("# Protect yourself from the internet...")
st.write(
    "## :shield: Try uploading an image protect it from generative AI :grin:"
)
# st.sidebar.write("## Upload and download :gear:")

# st.sidebar.markdown("\n")
# st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

image_buffers = st.file_uploader("#### Upload at least 2 images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
if image_buffers is None:
    pass
else:
    if len(image_buffers) < 2 and len(image_buffers) > 0:
        st.error("Please add at least 2 images")
    elif len(image_buffers) == 0:
        pass
    else:
        base64_imgs = []
        pil_imgs = []
        clicked = None
        for image_buffer in image_buffers:
            encoded = base64.b64encode(image_buffer.read()).decode()
            base64_imgs.append(f"data:image/jpeg;base64,{encoded}")
            pil_imgs.append(Image.open(image_buffer))
        st.session_state.image_buffers = base64_imgs

        c1, c2 = st.columns(2)
        with c1:
            clicked = clickable_images(
                base64_imgs,
                titles=[f"Image #{str(i)}" for i in range(len(base64_imgs))],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                img_style={"margin": "5px", "height": "300px", "width": "300px"},
            )
        with c2:
            pass

        if clicked is not None:
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            with gc2:
                infer = st.button("Protect me!")
                if infer:
                    print("hehehehe")

