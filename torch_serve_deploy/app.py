from fastapi import FastAPI, HTTPException

app = FastAPI()
from autoencoder.models import variational_model, vanilla_model
model = variational_model(
    input_shape=(96, 96, 3), 
    latent_dim=16,
    )

@app.get("/random")
def random():
   pass 
