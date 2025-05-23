import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from generator_model import Generator  # Make sure this file is in the same directory
from generator_model import Generator  

# Streamlit page config
st.set_page_config(page_title="DCGAN Digit Generator", layout="centered")
st.title("🧠 DCGAN - Handwritten Digit Generator")

# Load model
@st.cache_resource
def load_model():
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location=torch.device('cpu')))
    model.load_state_dict(torch.load("./Models/generator.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

G = load_model()
latent_dim = 100

st.sidebar.header("Options")
mode = st.sidebar.radio("Choose Mode", ("Generate Digits", "Retrain Generator"))

if mode == "Generate Digits":
    num_images = st.sidebar.slider("Number of digits to generate", min_value=4, max_value=64, value=16, step=4)

    if st.button("Generate Digits"):
        with st.spinner("Generating images..."):
            noise = torch.randn(num_images, latent_dim)
            fake_images = G(noise).detach().cpu()

            # Plot generated digits
            grid = vutils.make_grid(fake_images, nrow=8, normalize=True)
            plt.figure(figsize=(10, 10))
            plt.axis("off")
            plt.imshow(grid.permute(1, 2, 0).numpy())
            st.pyplot(plt)

elif mode == "Retrain Generator":
    st.markdown("### Upload a new image to retrain the generator (demo purpose only)")
    uploaded_file = st.file_uploader("Upload a 28x28 grayscale image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Uploaded Image", width=150)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        real_image = transform(image).unsqueeze(0)  # Add batch dimension
        real_image = real_image.view(-1, 784)

        st.markdown("Retraining for 1 step to adapt (for demo only)...")

        # Minimal retraining demo
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

        noise = torch.randn(1, latent_dim)
        fake_image = G(noise)
        fake_image_flat = fake_image.view(-1, 784)

        target = torch.ones_like(fake_image_flat)
        loss = criterion(fake_image_flat, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st.success("Retraining step complete. Generated image after one step:")

        fake_image = G(noise).detach()
        grid = vutils.make_grid(fake_image, normalize=True)
        plt.figure(figsize=(3, 3))
        plt.axis("off")
        plt.imshow(grid.permute(1, 2, 0).squeeze().numpy(), cmap="gray")
        st.pyplot(plt)

st.markdown("""
---
Made with ❤️ using PyTorch and Streamlit.  
Upload your `generator_model.py` and `generator.pth` to run this app locally.
""")
# Note: This is a simplified example. In a real-world scenario, you would want to implement proper retraining logic and handle the model's state more robustly.
# Ensure you have the required libraries installed:
# pip install streamlit torch torchvision matplotlib pillow
# To run the app, save this code in a file named `app.py` and run:
# streamlit run app.py
# Make sure to have the generator_model.py file in the same directory with the Generator class defined.
# The generator.pth file should contain the trained model weights.