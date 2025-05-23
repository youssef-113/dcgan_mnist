import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import torchvision.transforms as transforms
from generator_model import Generator  # Ensure this file defines your Generator

# ---------------------
# Streamlit App Config
# ---------------------
st.set_page_config(
    page_title="DCGAN Handwritten Digit Generator",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 DCGAN - Handwritten Digit Generator")
st.write(
    "This app uses a trained DCGAN generator to create realistic-looking handwritten digits. "
    "Adjust parameters below and generate new samples on the fly!"
)

# ---------------------
# Load and Cache Model
# ---------------------
@st.cache_resource
def load_generator_model(path: str = "./Models/generator.pth") -> torch.nn.Module:
    model = Generator()
    try:
        state = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state)
        model.eval()
    except FileNotFoundError:
        st.error(f"Model file not found at {path}. Please ensure the path is correct.")
        st.stop()
    return model

G = load_generator_model()

# ---------------------
# Sidebar Controls
# ---------------------
st.sidebar.header("Generation Settings")

num_cols = st.sidebar.slider(
    label="Columns",
    min_value=4,
    max_value=16,
    value=8,
    step=1,
    help="Number of columns in the generated image grid."
)
num_rows = st.sidebar.slider(
    label="Rows",
    min_value=1,
    max_value=8,
    value=2,
    step=1,
    help="Number of rows in the generated image grid."
)
noise_dim = st.sidebar.number_input(
    label="Latent Vector Size",
    min_value=10,
    max_value=200,
    value=100,
    step=10
)
seed = st.sidebar.number_input(
    label="Random Seed (optional)",
    min_value=0,
    value=42,
    step=1
)

if st.sidebar.button("Generate Digits"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_images = num_cols * num_rows
    noise = torch.randn(total_images, noise_dim)

    with st.spinner("Generating images..."):
        fake_images = G(noise).detach().cpu()

    grid = vutils.make_grid(
        fake_images,
        nrow=num_cols,
        normalize=True,
        scale_each=True
    )

    grid_np = grid.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(figsize=(num_cols, num_rows))
    ax.axis("off")
    ax.imshow(grid_np)
    st.pyplot(fig)

    buf = BytesIO()
    plt.imsave(buf, grid_np)
    buf.seek(0)
    st.download_button(
        label="Download Image",
        data=buf,
        file_name="dcgan_digits.png",
        mime="image/png"
    )

# ---------------------
# Optional: Retraining Demo (Hidden)
# ---------------------
st.sidebar.header("Demo: Single-Step Retrain")
retrain = st.sidebar.checkbox(
    label="Enable single-step retraining",
    help="For demonstration only: perform one gradient step on a single uploaded image."
)
if retrain:
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        label="Upload 28×28 grayscale image", type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None and st.sidebar.button("Retrain Generator"):
        # Load and display uploaded image via Matplotlib to avoid Streamlit image errors
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        fig1, ax1 = plt.subplots()
        ax1.axis('off')
        ax1.imshow(image, cmap='gray')
        st.pyplot(fig1)

        # Prepare input tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        real = transform(image).unsqueeze(0)

        # One-step update demonstration
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(G.parameters(), lr=2e-4)
        noise_vec = torch.randn(1, noise_dim)
        fake = G(noise_vec)

        loss = criterion(fake.view(-1), torch.ones_like(fake.view(-1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st.success("Retraining step completed.")
        st.write(f"Loss: {loss.item():.4f}")

        # Display updated generated image
        updated = G(noise_vec).detach().cpu().view(28, 28)
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.imshow(updated, cmap='gray')
        st.pyplot(fig2)

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.write(
    "Ensure you have the required libraries installed: `pip install streamlit torch torchvision matplotlib pillow`."
)
st.write(
    "To run: `streamlit run app.py`. Place `generator_model.py` and `Models/generator.pth` in the same directory."
)