from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from model import GAN
import matplotlib.pyplot as plt
import time

height = 64
width = 64
channels = 3
latent_dim = 100

model = GAN()
gan = model.generate_GAN()
discriminator = model.discriminator
generator = model.generator
#gan.load('models/gan.h5')

x_train = []
#datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.05,height_shift_range=0.05,shear_range=0.1,zoom_range=0.1,horizontal_flip=True,fill_mode='nearest')
for i in range(1262):
    img = image.load_img('faces64/face'+str(i)+'.jpg')
    img = image.img_to_array(img)
    x_train.append(img)
    #img = img.reshape((1, ) + (height, width, channels)).astype('float32')/255.
    '''c = 0
    for batch in datagen.flow(img, batch_size=1):
        #plt.figure(c)
        #plt.imshow(image.array_to_img(batch[0]))
        x_train.append(batch[0])
        if c==30:
            break
        c+=1'''


x_train = np.array(x_train)
x_train = x_train.reshape(
    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.
np.random.shuffle(x_train)
print('x_train.shape: ', x_train.shape)

iterations = 700000
batch_size = 32
save_dir = './gan_images/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

start = 0
sta = time.time()

for step in range(1262*25):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)
    
    start += batch_size
    if start > len(x_train) - batch_size:
      start = 0

    if step % 100 == 0:
        gan.save('models/gan15.h5')

        print('스텝 %s에서 판별자 손실: %s' % (step, d_loss))
        print('스텝 %s에서 적대적 손실: %s' % (step, a_loss))

        img = image.array_to_img(generated_images[0]*255., scale=False)
        img.save(os.path.join(save_dir, 'model/generated_images/generated_image' + str(step) + '.jpg'))
        
        img = image.array_to_img(real_images[0]*255., scale=False)
        img.save(os.path.join(save_dir, 'model/real_images/real_image' + str(step) + '.jpg'))
        print(time.time()-sta)
        
random_latent_vectors = np.random.normal(size=(10, latent_dim))

generated_images = generator.predict(random_latent_vectors)

for i in range(generated_images.shape[0]):
    img = image.array_to_img(generated_images[i] * 255., scale=False)
    plt.figure()
    plt.imshow(img)
    
plt.show()