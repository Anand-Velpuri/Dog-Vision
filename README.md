# Dog Vision 🐕

A modern web application that uses deep learning to classify dog breeds from uploaded images. Built with Flask, TensorFlow, and a beautiful responsive UI.

## Features ✨

- **Instant Dog Breed Classification**: Upload an image and get predictions in real-time
- **Modern UI/UX**: Clean, responsive design with smooth animations
- **Dark/Light Mode**: Toggle between themes for comfortable viewing
- **Recent Predictions**: View and manage your last 5 predictions
- **Progress Indicators**: Beautiful circular progress bar with animations
- **Mobile Responsive**: Works seamlessly on all devices

## Tech Stack 🛠️

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, TensorFlow Hub (MobileNetV2)
- **Frontend**: HTML, CSS, JavaScript
- **Database**: IndexedDB for client-side storage
- **UI Components**: Custom CSS with modern design patterns

## Setup 🚀

1. Clone the repository:

```bash
git clone https://github.com/yourusername/dog-vision.git
cd dog-vision
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage 💡

1. Click on the upload area or drag and drop an image
2. Wait for the upload animation to complete
3. Click "Predict" to get the dog breed classification
4. View the prediction result with confidence score
5. Check your recent predictions in the gallery below
6. Toggle dark/light mode using the theme switcher

## ⚠️ Important Note (Woof!) 🐾

Hey hooman! This AI is trained exclusively on dog images (10,000+ training images and 10,000+ validation images of our furry friends). So please:

- ✅ DO upload pictures of dogs (they make me happy!)
- ❌ DON'T upload pictures of humans (I might think you're a Poodle or a Husky 😅)
- ❌ DON'T upload pictures of cats (I'll be very confused and might need therapy 🐱)
- ❌ DON'T upload other random stuff (I only speak dog, sorry!)

Remember: I'm a good boy who only knows dogs! 🐕

## Project Structure 📁

```
dog-vision/
├── app.py              # Flask application
├── static/
│   ├── css/
│   │   └── styles.css  # Styling
│   └── js/
│       └── script.js   # Frontend logic
├── templates/
│   └── index.html      # Main page
├── model/              # ML model directory
├── data/
│   └── labels.csv      # Breed labels
└── requirements.txt    # Python dependencies
```

## Model Details 🧠

The application uses MobileNetV2 from TensorFlow Hub, fine-tuned on a dog breed dataset. The model can classify 120 unique dog breeds with high accuracy while maintaining fast inference times.

### Performance Metrics 📊

- **Accuracy**: 99.92% (0.9992)
- **Loss**: 0.0091
- **Training Data**: 10,000+ images
- **Validation Data**: 10,000+ images

These impressive metrics demonstrate the model's exceptional ability to correctly identify dog breeds. However, remember that this accuracy is specific to dog images - the model is not trained to recognize any other types of images! 🐕

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Made with ❤️ by Anand Velpuri

If you like this project, please give it a ⭐️!
