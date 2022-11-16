from flask import Flask, render_template, url_for, request

from fonction import predict

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"]=1

@app.route("/")
def index():
    a= 25 
    b="La deep nous a quitter "

    return render_template("index.html", age=a, déclaration=b)


@app.route("/Formulaire")
def formulaire():
    a= 25 
    b="La deep nous a quitter "

    return render_template("formulaire.html", age=a, déclaration=b)

@app.route("/Formulaire/reponse", methods = ["post"])
def fichier():
    u=request.files["iphot"]
    t=predict(u)
    
    print(t)

    return render_template ("formulaire.html", v=t)


if __name__ == "__main__" :
    app.run(debug =True)