from flask import Flask, render_template, request
import base64
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model('malaria_detection_model.h5')

# Define the Flask app
app = Flask(__name__)

# Define a function to preprocess the image
def preprocess_image(img):
    img = img.resize((64, 64))
    img = img.convert('RGB')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the POST request
    img = request.files['file'].read()
    img = image.load_img(BytesIO(img), target_size=(64,64))
    
    # Preprocess the image
    img = preprocess_image(img)
    
    # Get the model's prediction
    pred = model.predict(img)
    
    label = 'Infected' if pred > 0.5 else 'Not infected'

    
    # Encode the image as a Base64 string for display
    img_data = base64.b64encode(request.files['file'].read()).decode('ascii')
    
    # Render the prediction and image on the page
    return render_template('index.html', label=label, image_data=img_data)



if __name__ == '__main__':
    # Launch the browser automatically
    import webbrowser
    webbrowser.open_new('http://localhost:5000/')
    # Run the app
    app.run(debug=True)
