# Perception Sudoku

A unique Sudoku game where the clue numbers are predictions from a Convolutional Neural Network (CNN). Your goal: **catch the AI hallucinations!**

## The Concept

Normal Sudoku is a game of logic. **Perception Sudoku** is a game of perception.

We don't give you the numbers directly. Instead, we show you photos of numbers, and an AI tries to read them for you. But AI makes mistakes — sometimes it "hallucinates" and predicts the wrong digit. Can you spot when the AI is wrong?

## Features

- **CNN-Powered Predictions**: Each clue cell displays an image from the SVHN dataset with the model's predicted digit
- **Confidence Indicators**: Color-coded predictions (green = high confidence, yellow = medium, red = low)
- **Override System**: Click the pencil icon to correct AI mistakes
- **Uncertainty View**: Toggle to see the full softmax probability distribution
- **Debug Mode**: Reveal ground truth values to verify AI accuracy
- **Solution Checker**: Identifies errors and specifically calls out AI hallucinations

## Screenshots

The game displays a 9x9 Sudoku grid where:
- **Clue cells** show digit images with AI predictions
- **Empty cells** are standard input fields for you to fill
- Bold separators clearly mark the 3x3 boxes

## Installation

```bash
# Clone the repository
git clone https://github.com/wiswis01/sudo-kook.git
cd sudo-kook

# Install dependencies
pip install -r requirements.txt
```

## Required Data Files

You need to provide two files (not included due to size):

1. **`sudoku_cnn.h5`** - Trained CNN model
   - Input shape: `(32, 32, 1)` (grayscale)
   - Output: 10 classes (digits 0-9)

2. **`svhn_data.h5`** (optional) - SVHN digit images
   - If missing, the app generates synthetic noise images

Place both files in the project root directory.

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## How to Play

1. **Examine the clues**: Each clue cell shows an image and the AI's prediction
2. **Check confidence**: Green = confident, Red = uncertain
3. **Spot hallucinations**: If the image doesn't match the predicted number, override it!
4. **Fill empty cells**: Complete the Sudoku using standard rules
5. **Check solution**: Click "Check Solution" to see if you caught all the AI mistakes

## Project Structure

```
sudo-kook/
├── app.py              # Streamlit game interface
├── sudoku_utils.py     # Sudoku generation, image handling, validation
├── requirements.txt    # Python dependencies
├── sudoku_cnn.h5       # CNN model (not in repo)
└── svhn_data.h5        # SVHN images (not in repo)
```

## Technical Details

- **Sudoku Generation**: Backtracking algorithm generates valid puzzles with ~40 clues
- **Image Preprocessing**: Resizes to 32x32, converts to grayscale, normalizes to [0,1]
- **Model Input**: `np.expand_dims` ensures shape `(1, 32, 32, 1)`
- **State Management**: Uses `st.session_state` for persistence across interactions

## Dependencies

- streamlit
- tensorflow
- numpy
- h5py
- Pillow
- opencv-python-headless

## License

MIT
