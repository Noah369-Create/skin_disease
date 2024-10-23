from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model('skin_disease_model.h5')

def prepare_image(img_path):
    img = Image.open(img_path)
    img = img.resize((img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        file_path = './static/' + file.filename
        file.save(file_path)
        
        img = prepare_image(file_path)
        prediction = model.predict(img)
        predicted_class = categories[np.argmax(prediction)]
        
        return render_template('result.html', prediction=predicted_class, img_path=file_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
