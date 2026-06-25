# AI-Based Air Drawing

An innovative computer vision application that enables users to draw in the air using hand gestures. This project combines real-time hand tracking with AI-powered shape and letter recognition to create a seamless air drawing experience.

## Features

- **Air Drawing**: Draw in the air using just your index finger
- **Hand Gesture Control**: 
  - Index finger up: Draw mode
  - Two fingers (index + middle): Selection mode for UI interaction
- **Rich Toolset**:
  - 12-color palette with visual swatches
  - Pen and eraser tools
  - Adjustable brush size (2-100px)
  - Clear canvas functionality
  - Save drawings as PNG images
- **AI-Powered Recognition**:
  - Automatic shape correction (lines, circles, rectangles, squares, triangles)
  - Letter recognition (uppercase A-Z)
  - Real-time confidence feedback
- **Responsive UI**: Adaptive layout that works on different screen sizes
- **Smooth Drawing**: Advanced smoothing algorithm for natural drawing experience

## How It Works

1. **Hand Detection**: Uses MediaPipe Hand Landmarker to detect and track hand landmarks in real-time
2. **Gesture Recognition**: Identifies drawing and selection gestures based on finger positions
3. **Stroke Capture**: Records the path of your index finger as you draw
4. **AI Analysis**: When you finish a stroke, the system:
   - Attempts to recognize letters using a neural network
   - If no letter is found, analyzes the shape geometrically
   - Automatically corrects messy strokes into clean shapes (if enabled)
5. **Rendering**: Displays your drawing merged with the camera feed on a canvas

## Installation

### Prerequisites

- Python 3.8+
- Webcam
- Windows/Linux/macOS

### Setup

1. Clone the repository:
```bash
git clone https://github.com/1dpthr/Ai-Based-Air-Drawing.git
cd Ai-Based-Air-Drawing
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the hand landmarker model (automatic on first run):
   - The model will be downloaded automatically to `hand_landmarker.task` on first execution

## Usage

Run the main application:
```bash
python main.py
```

### Controls

- **Index Finger**: Raise only your index finger to draw
- **Two Fingers**: Raise index + middle fingers to select UI elements
- **S Key**: Save your drawing as `drawing.png`
- **Q Key**: Quit the application
- **A Key**: Toggle auto-shape correction on/off

### UI Elements

- **Colors Panel**: Click on color swatches to change pen color
- **Tools Panel**: 
  - PEN: Drawing mode
  - ERASER: Erase parts of your drawing
  - CLEAR: Clear the entire canvas
  - SAVE: Save drawing to file
- **Brush Size**: Use +/- buttons to adjust stroke thickness
- **Status Bar**: Shows current mode and system status

## Project Structure

```
Ai-Based-Air-Drawing/
├── main.py                      # Main application entry point
├── hand_tracking_module.py      # MediaPipe hand detection and gesture recognition
├── shape_recognition_module.py  # Geometric shape detection and correction
├── letter_recognition_module.py # Neural network-based letter recognition
├── train_emnist_uppercase.py    # Training script for letter recognition model
├── requirements-train.txt       # Dependencies for training
├── hand_landmarker.task         # MediaPipe hand landmarker model
└── README.md                    # This file
```

## Dependencies

### Runtime
- OpenCV (cv2)
- MediaPipe
- NumPy

### Training (optional)
- TensorFlow
- scikit-learn
- Pillow

## Training the Letter Recognition Model

If you want to train or retrain the letter recognition model:

1. Install training dependencies:
```bash
pip install -r requirements-train.txt
```

2. Download the EMNIST dataset (uppercase letters)

3. Run the training script:
```bash
python train_emnist_uppercase.py
```

## How It Works - Technical Details

### Hand Tracking
The system uses MediaPipe's Hand Landmarker model to detect 21 hand landmarks in real-time. The model runs on a downscaled version of the frame (480px width) for optimal performance, then maps coordinates to the display resolution.

### Gesture Detection
- **Drawing Pose**: Index finger extended, other fingers curled
- **Selection Pose**: Index and middle fingers extended, ring and pinky curled

### Shape Recognition
Uses geometric analysis including:
- Contour fitting and approximation
- PCA line fitting for straight lines
- Circularity analysis for circles
- Aspect ratio analysis for rectangles/squares
- Vertex counting for triangles

### Letter Recognition
Employs a neural network trained on the EMNIST dataset to recognize uppercase letters (A-Z) from stroke images.

## Performance Tips

- Ensure good lighting for better hand detection
- Keep your hand within the camera frame
- Use a solid-colored background for best results
- Adjust `SMOOTHING` parameter in `main.py` (0.0-1.0) to control drawing lag vs. responsiveness

## Troubleshooting

**Hand not detected?**
- Check lighting conditions
- Ensure your hand is fully visible in the frame
- Try adjusting `min_detection_confidence` in `hand_tracking_module.py`

**Drawing is laggy?**
- Lower the camera resolution in `main.py` (CAM_W, CAM_H)
- Close other applications using the camera
- Ensure your system meets the minimum requirements

**Model download fails?**
- Manually download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
- Place it in the project root directory

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand tracking technology
- [OpenCV](https://opencv.org/) for computer vision utilities
- [EMNIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset) dataset for letter recognition training

## Release Notes

### v1.0.0 (Initial Release)
- Real-time hand tracking and gesture recognition
- Air drawing with 12-color palette
- Shape auto-correction (line, circle, rectangle, square, triangle)
- Letter recognition (A-Z)
- Adjustable brush size
- Save and clear functionality
- Responsive UI design

---

**Note**: This project requires a webcam and works best in well-lit environments. The hand tracking model file (`hand_landmarker.task`) is approximately 7.8MB and will be downloaded automatically on first run.