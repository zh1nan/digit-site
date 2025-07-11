from flask import Flask, render_template, request, jsonify
import base64, io, torch
from PIL import Image
from torchvision import transforms
from acgan_models import Discriminator

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Discriminator().to(device)
model.load_state_dict(torch.load('acgan_discriminator.pth', map_location=device))
model.eval()

tfm = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((28, 28)),
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

def preprocess(data_url: str):
    header, b64 = data_url.split(',', 1)
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('L')
    return tfm(img).unsqueeze(0).to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    x = preprocess(request.json['image'])
    with torch.no_grad():
        _, logit = model(x)
        digit = torch.argmax(logit, 1).item()
    return jsonify({'digit': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
