from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from skimage.transform import resize
import imageio.v3 as iio
import numpy as np
import re
import base64

from fonction import predict

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/Upload")
def formulaire():
    return render_template("formulaire.html")


@app.route("/Formulaire/reponse", methods=["post"])
def fichier():
    u = request.files["iphot"]
    t = predict(u)

    print(t)

    return render_template("formulaire.html", v=t)


@app.route("/Dessin")
def dessin():
    return render_template("dessin.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict_dessin():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = iio.imread('output.png', mode='L')
    x = np.invert(x)
    x = resize(x, (28, 28))

    # reshape image data for use in neural network
    x = x.reshape(1, 28, 28, 1)

    model = load_model("model.h5")

    # with graph.as_default():
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    response = np.array_str(np.argmax(out, axis=1))
    return response


def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.decodebytes(imgstr))

if __name__ == "__main__":
    app.run(debug=True)
