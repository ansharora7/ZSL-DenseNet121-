--beta-map 0.01 \
--beta-con 0.01 \
--neg-penalty 0.50 \
lr=1e-3
optimizer=Adam
epochs=40
total_loss+=0.003*vae_loss
In loss it is not 0.0*aligmnent_loss