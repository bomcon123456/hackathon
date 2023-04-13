import streamlit as st
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import torch
import base64
from torchvision import transforms
from anti_dreambooth_wrapper import init_model, run
from st_clickable_images import clickable_images

IMAGE_SIZE = 512
ImageFile.LOAD_TRUNCATED_IMAGES = True
st.set_page_config(layout="wide", page_title="Gennaissance")


@st.cache_data(persist="disk", show_spinner=True)
def getmodels():
    return init_model()


f, tokenizer, noise_scheduler, vae = getmodels()


def get_prompt(ishuman):
    if ishuman:
        return "a photo of sks person", "a photo of person"
    # TODO: do for art
    return "a photo of sks person", "a photo of person"


image_transforms = transforms.Compose(
    [
        transforms.Resize(
            IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

if "image_buffers" not in st.session_state:
    st.session_state.image_buffers = None


footer = """<style>
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
z-index: 9999;
}
</style>
<div class="footer">
<p>Developed with ❤ by Gennaissance. We will never store your data on our server. Your privacy matters.</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

st.write("# Protect yourself from the internet...")
st.write("## :shield: Try uploading an image protect it from generative AI :grin:")
# st.sidebar.write("## Upload and download :gear:")

# st.sidebar.markdown("\n")
# st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

ishuman = st.checkbox("Protect human faces")
image_buffers = st.file_uploader(
    "#### Upload at least 2 images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
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
            img_ = Image.open(image_buffer).convert("RGB")
            img = image_transforms(img_)
            pil_imgs.append(img)
        pil_imgs = torch.stack(pil_imgs)
        st.session_state.image_buffers = base64_imgs

        c1, c2 = st.columns(2)
        with c1:
            clicked = clickable_images(
                base64_imgs,
                titles=[f"Image #{str(i)}" for i in range(len(base64_imgs))],
                div_style={
                    "display": "flex",
                    "justify-content": "center",
                    "flex-wrap": "wrap",
                },
                img_style={"margin": "5px", "height": "300px", "width": "300px"},
                key="left",
            )
        b64_final_images = None
        if len(base64_imgs) > 0:
            gc1, gc2, gc3, gc4, gc5 = st.columns(5)
            with gc2:
                infer = st.button("Protect me!")
                if infer:
                    with st.spinner("Cloaking your images..."):
                        final_images = run(f, tokenizer, noise_scheduler, vae, pil_imgs)
                    b64_final_images = []
                    for fimg in final_images:
                        im_file = BytesIO()
                        fimg.save(im_file, format="JPEG")
                        im_bytes = (
                            im_file.getvalue()
                        )  # im_bytes: image in binary format.
                        im_b64 = base64.b64encode(im_bytes).decode("utf-8")
                        b64_final_images.append(f"data:image/jpeg;base64,{im_b64}")

        if b64_final_images is not None and c2 is not None:
            with c2:
                newclicked = clickable_images(
                    b64_final_images,
                    titles=[f"Image #{str(i)}" for i in range(len(b64_final_images))],
                    div_style={
                        "display": "flex",
                        "justify-content": "center",
                        "flex-wrap": "wrap",
                    },
                    img_style={"margin": "5px", "height": "300px", "width": "300px"},
                    key="right",
                )
