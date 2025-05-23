import streamlit as st
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
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
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state)
    model.eval()
    return model

G = load_generator_model()

# ---------------------
# Sidebar Controls
# ---------------------
st.sidebar.header("Generation Settings")

num_cols = st.sidebar.slider(
    "Columns", 4, 16, 8, 1,
    help="Number of columns in the generated image grid."
)
num_rows = st.sidebar.slider(
    "Rows", 1, 8, 2, 1,
    help="Number of rows in the generated image grid."
)
noise_dim = st.sidebar.number_input(
    "Latent Vector Size", min_value=10, max_value=200, value=100, step=10
)
seed = st.sidebar.number_input(
    "Random Seed (optional)", min_value=0, value=42, step=1
)

if st.sidebar.button("Generate Digits"):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_images = num_cols * num_rows
    noise = torch.randn(total_images, noise_dim)

    with st.spinner("Generating images..."):
        fake_images = G(noise).detach().cpu()

    # Create grid
    grid = vutils.make_grid(
        fake_images,
        nrow=num_cols,
        normalize=True,
        scale_each=True
    )

    # Convert to NumPy for plotting
    grid_np = grid.permute(1, 2, 0).numpy()

    # Display
    fig, ax = plt.subplots(figsize=(num_cols, num_rows))
    ax.axis("off")
    ax.imshow(grid_np)
    st.pyplot(fig)

    # Allow download
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
    "Enable single-step retraining",
    help="For demonstration only: perform one gradient step on a single uploaded image."
)
if retrain:
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload 28×28 grayscale image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_file is not None and st.sidebar.button("Retrain Generator"):
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Uploaded Image", width=100)

        # Prepare input
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        real = transform(image).unsqueeze(0)

        # One-step update
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(G.parameters(), lr=2e-4)
        noise = torch.randn(1, noise_dim)
        fake = G(noise)

        loss = criterion(fake.view(-1), torch.ones_like(fake.view(-1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st.success("Retraining step completed.")
        st.write(f"Loss: {loss.item():.4f}")

        # Show updated sample
        updated = G(noise).detach().cpu()
        fig2, ax2 = plt.subplots()
        ax2.axis('off')
        ax2.imshow(updated.view(28, 28), cmap='gray')
        st.pyplot(fig2)

# ---------------------
# Footer
# ---------------------
st.markdown(
    "---"
)
st.write(
    "Ensure you have the required libraries installed: `pip install streamlit torch torchvision matplotlib pillow`."
)
st.write(
    "To run: `streamlit run app.py`. Place `generator_model.py` and `Models/generator.pth` in the same directory."
)
