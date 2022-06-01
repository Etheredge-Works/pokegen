import streamlit as st
import dvc.api
from autoencoder import models
import yaml
import torch

with dvc.api.open("models/vae_conv_kwargs.yaml") as f:
    kwargs = yaml.load(f, Loader=yaml.Loader)

def app():
    st.title("Pokegen Demo Site")
    st.write("This is a demo site for my experiment in generative AI through Pokemon datasets.")
    st.write("Testing out the Streamlit library.")

    # st.write(model_url)
    st.write(kwargs)
    model = models.VAE(**kwargs)
    model.load_state_dict(torch.load("models/vae_conv.pt"))
    st.write(model)
    with dvc.api.open(
            "models/vae_conv.pt",
            repo='https://github.com/etheredge-works/pokegen', 
            mode='rb',
            # encoding="bytes"
            ) as model_file:
        st.write(model_file)
        torch_loaded = torch.load(model_file)
        # st.write(torch_loaded)
        model.load_state_dict(torch_loaded)


    #vae = models.VAE(*


app()