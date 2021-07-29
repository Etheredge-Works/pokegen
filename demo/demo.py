import streamlit as st
import ae
import vae


PAGES = {
    "Autoencoder": ae,
    "Variational Autoencoder": vae
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()