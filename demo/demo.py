import streamlit as st
# import ae
# import vae
import dvc.api
from autoencoder import models
import torch
import yaml
import torchvision.transforms as T


model_paths = [
    ("ae", models.AutoEncoder, "models/ae_conv.pt", "models/ae_conv_kwargs.yaml"),
    ("vae", models.VAE, "models/vae_conv.pt", "models/vae_conv_kwargs.yaml"),
    
]
models_dict = {}

for key, model_const, model_path, kwargs_path in model_paths:
    with dvc.api.open(
        kwargs_path,
        repo='https://github.com/etheredge-works/pokegen', 
        mode='rb',
        ) as f:
        kwargs = yaml.load(f, Loader=yaml.Loader)
    model = model_const(**kwargs)
    # model.load_state_dict(torch.load("models/vae_conv.pt"))
    # st.write(model)
    with dvc.api.open(
            model_path,
            repo='https://github.com/etheredge-works/pokegen', 
            mode='rb',
            # encoding="bytes"
            ) as model_file:
        # st.write(model_file)
        torch_loaded = torch.load(model_file, map_location=torch.device('cpu'))
        # st.write(torch_loaded)
        model.load_state_dict(torch_loaded)
        # st.write(model)
    models_dict[key] = model






PAGES = {
    "Autoencoder": models_dict["ae"],
    "Variational Autoencoder": models_dict["vae"],
}
st.sidebar.title('Model Selection')
st.title("Pokegen Demo Site")
st.write("This is a simple demo site for my experiment in generative AI through Pokemon datasets.")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
model = PAGES[selection]
button = st.button("Shuffle Noise")
noise = torch.zeros(1, model.latent_size)
if button:
    noise = torch.randn(1, model.latent_size)
with st.expander("Noise for Generation"):
    for i in range(model.latent_size):
        # st.write(f"{i}:", noise[:, i])
        item = st.sidebar.slider(f"Noise-{i}", -5.0, 5.0, float(noise[0, i]))
        noise[0, i] = item
        
    # noise = st.sidebar.slider("Noise", 0, 1, 0.5)
    x_hat = model.generate(noise)
    image = T.ToPILImage()(x_hat.cpu().detach().squeeze(0)).resize((256, 256))
    # image = T.ToPILImage()(x_hat.cpu().detach().squeeze(0).permute(1, 2, 0))
st.image(image)