from tensorflow.keras.models import load_model
from PIL import Image
# from utils import imread, imresize
import imageio
# imageio.imread("xxxx.png")
import numpy as np
model = load_model("model_hand.h5")


def predict(image):
    print('debut')
    x = Image.open(image).convert("L")
    print("milieu")
    x = x.resize((28,28))

    x_np = np.array(x, order='C')
    x_reshaped = x_np.reshape(((1,28,28,1)))

    print('b')
    # print(x.shape)
    # reshape image data for use in neural network
    # x = x.reshape(1,28,28,1)
    out = model.predict(x_reshaped)
    print("fin")
    print(out)
    print(np.argmax(out, axis=1))
    
    

    # resultat=model.predict([])

    return np.argmax(out, axis=1)