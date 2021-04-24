import os
import joblib
import test

from flask import Flask, request, render_template, send_from_directory

__author__ = 'AlphaGeeks'

app = Flask(__name__)


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    # print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    # print(request.files.getlist("file"))

    image_names = []

    for upload in request.files.getlist("file"):
        # print(upload)
        #print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        #print ("Accept incoming file:", filename)
        #print ("Save it to:", destination)
        image_names.append(destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    if request.method == 'POST':
        height = request.form['height']
        leg_length = request.form['leg_length']
        gender = request.form['gender']

        # Check Input information at Terminal

        print("\n\nFront body image: " + image_names[0])
        print("Front body image with arms open: " + image_names[1])
        print("Side body image: " + image_names[2])
        print("Height " + height)
        print("Leg length " + leg_length)
        print("Gender " + gender)
        print("\n\n")

        print("Estimating size...\n")

        try:
            out = test.initials(float(height), float(
                leg_length), gender, image_names[0], image_names[1], image_names[2])
        except Exception as e:
            print(e)
            return render_template("complete.html", size_us_d = '', size_eu_d = '', size_us_t = '', size_eu_t = '', size_us_j = '', size_eu_j = '')

        a = str(out[0])
        b = str(out[1])
        c = str(out[2])
        d = str(out[3])
				e = str(out[4])
				f = str(out[5])
        return render_template("complete.html", size_us_d = a, size_eu_d = b, size_us_t = c, size_eu_t = d, size_us_j = e, size_eu_j = f)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
