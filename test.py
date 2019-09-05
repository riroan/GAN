from model import GAN
import numpy as np
from keras.preprocessing import image

latent_dim = 100
num_of_generate_images=500

z = np.random.normal(size = (num_of_generate_images,latent_dim))

weight_path = ''                                              # fill it!
model = GAN()
gan = model.generate_GAN()
model.load(weight_path)

images = model.generator.predict(z)

cnt = 0
for i in images:
    img = image.array_to_img(i)
    img_path = ''                                              # fill it!
    img.save(img_path)
    cnt+=1
    