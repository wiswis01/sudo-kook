"""
Perception Sudoku
-----------------
A playable web game where initial numbers are CNN predictions,
allowing users to catch "AI Hallucinations."
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, List, Optional
import os

from sudoku_utils import (
    SudokuGenerator,
    ImageFetcher,
    ImagePreprocessor,
    check_solution
)

# Page configuration
st.set_page_config(
    page_title="Perception Sudoku",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better grid styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        padding: 2px;
        height: 35px !important;
        min-height: 35px !important;
    }
    .prediction-text {
        text-align: center;
        font-size: 14px;
        font-weight: bold;
        margin: 0;
        line-height: 1.2;
    }
    div[data-testid="column"] {
        padding: 1px !important;
    }
    .hallucination-alert {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .success-message {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    /* Compact buttons */
    .stButton > button {
        padding: 0px 8px;
        font-size: 12px;
        min-height: 25px;
        height: 25px;
    }
    /* Smaller progress bar */
    .stProgress > div > div {
        height: 4px !important;
    }
    /* Compact number input */
    .stNumberInput > div > div > input {
        padding: 2px;
        font-size: 12px;
    }
    /* Box separators */
    .box-separator-right {
        border-right: 3px solid #333 !important;
        padding-right: 8px !important;
    }
    .box-separator-bottom {
        border-bottom: 3px solid #333 !important;
        padding-bottom: 8px !important;
        margin-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the CNN model with caching."""
    model_path = "sudoku_cnn.h5"
    if not os.path.exists(model_path):
        st.error(f"Model '{model_path}' not found. Please place the model in the root directory.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def get_image_fetcher():
    """Initialize and cache the image fetcher."""
    return ImageFetcher()


def initialize_game():
    """Initialize a new game state."""
    # Generate puzzle
    generator = SudokuGenerator()
    puzzle, solution, clue_positions = generator.generate_puzzle(num_clues=40)

    # Get image fetcher
    fetcher = get_image_fetcher()
    model = load_model()

    # Store data for clue cells
    clue_data = {}
    predictions = {}
    model_stats = {
        'total_predictions': 0,
        'correct_predictions': 0,
        'confidence_sum': 0,
        'hallucinations': []
    }

    for row, col in clue_positions:
        ground_truth = solution[row][col]

        # Fetch image for this digit
        original_image = fetcher.get_image_for_digit(ground_truth)

        # Preprocess for model
        preprocessed = ImagePreprocessor.preprocess_for_model(original_image)

        # Get prediction
        predicted_digit = ground_truth  # Default fallback
        confidence = 1.0 
        softmax_probs = np.zeros(10)
        softmax_probs[ground_truth] = 1.0

        if model is not None:
            try:
                prediction = model.predict(preprocessed, verbose=0)

                # Handle different output shapes
                if prediction.ndim == 1:
                    softmax_probs = prediction
                else:
                    softmax_probs = prediction[0]

                # Ensure we have 10 classes (0-9)
                if len(softmax_probs) >= 10:
                    predicted_digit = int(np.argmax(softmax_probs[:10]))
                    confidence = float(softmax_probs[predicted_digit])
                else:
                    # If model outputs fewer classes, adjust
                    predicted_digit = int(np.argmax(softmax_probs)) + 1  # Shift if 1-9
                    if predicted_digit > 9:
                        predicted_digit = 9
                    confidence = float(np.max(softmax_probs))

                # Handle case where model predicts 0 (might need to map to 1-9)
                if predicted_digit == 0:
                    # My model uses 0-9
                    # For Sudoku we need 1-9
                    probs_1_to_9 = softmax_probs[1:10] if len(softmax_probs) >= 10 else softmax_probs
                    predicted_digit = int(np.argmax(probs_1_to_9)) + 1
                    confidence = float(probs_1_to_9[predicted_digit - 1]) if predicted_digit <= len(probs_1_to_9) else 0.5

            except Exception as e:
                st.warning(f"Prediction error at ({row}, {col}): {e}")
                predicted_digit = ground_truth
                confidence = 1.0

        # Update stats
        model_stats['total_predictions'] += 1
        model_stats['confidence_sum'] += confidence
        if predicted_digit == ground_truth:
            model_stats['correct_predictions'] += 1
        else:
            model_stats['hallucinations'].append({
                'row': row,
                'col': col,
                'predicted': predicted_digit,
                'actual': ground_truth,
                'confidence': confidence
            })

        # Prepare display image
        display_image = ImagePreprocessor.prepare_for_display(original_image)

        clue_data[(row, col)] = {
            'image': display_image,
            'ground_truth': ground_truth,
            'predicted': predicted_digit,
            'confidence': confidence,
            'softmax': softmax_probs[:10] if len(softmax_probs) >= 10 else softmax_probs,
            'overridden': False,
            'override_value': None
        }

        predictions[(row, col)] = predicted_digit

    return {
        'puzzle': puzzle,
        'solution': solution,
        'clue_positions': clue_positions,
        'clue_data': clue_data,
        'predictions': predictions,
        'user_inputs': {},
        'model_stats': model_stats,
        'game_complete': False,
        'using_synthetic': fetcher.using_synthetic
    }


def render_sidebar():
    """Render the sidebar with model stats and controls."""
    st.sidebar.title("🎮 Perception Sudoku")
    st.sidebar.markdown("---")

    # New Game button
    if st.sidebar.button("🔄 New Game"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")

    # Model Stats
    st.sidebar.subheader("Model Stats")

    if 'game_state' in st.session_state:
        stats = st.session_state.game_state['model_stats']

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Predictions", stats['total_predictions'])
        with col2:
            accuracy = (stats['correct_predictions'] / stats['total_predictions'] * 100
                        if stats['total_predictions'] > 0 else 0)
            st.metric("Accuracy", f"{accuracy:.1f}%")

        avg_confidence = (stats['confidence_sum'] / stats['total_predictions']
                          if stats['total_predictions'] > 0 else 0)
        st.sidebar.metric("Avg Confidence", f"{avg_confidence:.2%}")

        hallucination_count = len(stats['hallucinations'])
        st.sidebar.metric("🔴 Hallucinations", hallucination_count)

        if st.session_state.game_state['using_synthetic']:
            st.sidebar.warning("⚠️ Using synthetic noise images (SVHN data not found)")

    st.sidebar.markdown("---")

    # Confidence Toggle
    st.sidebar.subheader("🔧 Display Options")
    show_uncertainty = st.sidebar.checkbox(
        "Show AI Uncertainty",
        value=st.session_state.get('show_uncertainty', False),
        help="Display softmax probability distribution for each clue"
    )
    st.session_state.show_uncertainty = show_uncertainty

    show_debug = st.sidebar.checkbox(
        "Debug Mode",
        value=st.session_state.get('show_debug', False),
        help="Show ground truth for clue cells"
    )
    st.session_state.show_debug = show_debug

    st.sidebar.markdown("---")

    # Instructions
    st.sidebar.subheader("📖 How to Play")
    st.sidebar.markdown("""
    1. **Clue cells** show images with AI predictions
    2. AI predictions **may be wrong** (hallucinations!)
    3. Click **Override** to correct AI mistakes
    4. Fill empty cells with your answers
    5. Click **Check Solution** when done

    **Goal:** Catch the AI hallucinations!
    """)


def render_confidence_bar(softmax_probs: np.ndarray, predicted: int):
    """Render a confidence visualization."""
    # Create a simple bar chart of probabilities
    probs = softmax_probs.copy()
    if len(probs) < 10:
        probs = np.pad(probs, (0, 10 - len(probs)))

    # Display as horizontal bars for digits 1-9
    chart_data = {str(i): float(probs[i]) for i in range(1, 10)}

    # Simple text-based visualization
    st.caption("Confidence Distribution:")
    for digit in range(1, 10):
        prob = probs[digit] if digit < len(probs) else 0
        bar_width = int(prob * 20)
        bar = "█" * bar_width + "░" * (20 - bar_width)
        color = "🟢" if digit == predicted else "⚪"
        st.caption(f"{color} {digit}: {bar} {prob:.1%}")


def render_cell(game, row, col, show_uncertainty, show_debug):
    """Render a single cell."""
    cell_key = (row, col)

    if cell_key in game['clue_data']:
        clue = game['clue_data'][cell_key]

        # Display the image (smaller)
        st.image(clue['image'], width=40)

        if show_uncertainty:
            render_confidence_bar(clue['softmax'], clue['predicted'])
        else:
            if clue['overridden'] and clue['override_value'] is not None:
                display_value = clue['override_value']
                st.markdown(f"<p class='prediction-text' style='color: blue;'>{display_value}</p>",
                            unsafe_allow_html=True)
            else:
                display_value = clue['predicted']
                confidence_color = "#00aa00" if clue['confidence'] > 0.8 else (
                    "#ffaa00" if clue['confidence'] > 0.5 else "#ff0000")
                st.markdown(
                    f"<p class='prediction-text' style='color: {confidence_color};'>{display_value}</p>",
                    unsafe_allow_html=True)

            st.progress(min(clue['confidence'], 1.0))

        if show_debug:
            st.caption(f"GT: {clue['ground_truth']}")

        override_key = f"override_{row}_{col}"
        if st.button("✏️", key=override_key, help="Override"):
            st.session_state[f"editing_{row}_{col}"] = True

        if st.session_state.get(f"editing_{row}_{col}", False):
            new_val = st.number_input(
                "Val", min_value=1, max_value=9, value=clue['predicted'],
                key=f"input_{row}_{col}_override", label_visibility="collapsed"
            )
            if st.button("✓", key=f"confirm_{row}_{col}"):
                game['clue_data'][cell_key]['overridden'] = True
                game['clue_data'][cell_key]['override_value'] = new_val
                game['predictions'][cell_key] = new_val
                st.session_state[f"editing_{row}_{col}"] = False
                st.rerun()
    else:
        current_val = game['user_inputs'].get(cell_key, "")
        user_input = st.text_input(
            f"Cell {row},{col}",
            value=str(current_val) if current_val else "",
            max_chars=1, key=f"cell_{row}_{col}",
            label_visibility="collapsed", placeholder="?"
        )
        if user_input:
            try:
                val = int(user_input)
                if 1 <= val <= 9:
                    game['user_inputs'][cell_key] = val
                else:
                    game['user_inputs'][cell_key] = None
            except ValueError:
                game['user_inputs'][cell_key] = None
        else:
            game['user_inputs'][cell_key] = None


def render_grid():
    """Render the Sudoku grid with 3x3 box separators."""
    if 'game_state' not in st.session_state:
        return

    game = st.session_state.game_state
    show_uncertainty = st.session_state.get('show_uncertainty', False)
    show_debug = st.session_state.get('show_debug', False)

    # Create grid with 3 major column groups (for vertical separators)
    # Using column ratios: [3 cells, separator, 3 cells, separator, 3 cells]
    for box_row in range(3):  # 3 horizontal box groups
        for local_row in range(3):  # 3 rows per box
            row = box_row * 3 + local_row

            # Create columns: 3 + separator + 3 + separator + 3
            col_spec = [1, 1, 1, 0.15, 1, 1, 1, 0.15, 1, 1, 1]
            cols = st.columns(col_spec)

            col_idx = 0
            for box_col in range(3):  # 3 vertical box groups
                for local_col in range(3):  # 3 columns per box
                    col = box_col * 3 + local_col
                    with cols[col_idx]:
                        render_cell(game, row, col, show_uncertainty, show_debug)
                    col_idx += 1

                # Add vertical separator column (dark line)
                if box_col < 2:
                    with cols[col_idx]:
                        st.markdown(
                            "<div style='background-color: #333; width: 3px; height: 100%; min-height: 120px;'></div>",
                            unsafe_allow_html=True
                        )
                    col_idx += 1

        # Add horizontal separator between box rows
        if box_row < 2:
            st.markdown("<hr style='border: none; border-top: 3px solid #333; margin: 8px 0;'>", unsafe_allow_html=True)


def check_user_solution():
    """Check the user's solution."""
    if 'game_state' not in st.session_state:
        return

    game = st.session_state.game_state

    # Build the user's grid
    user_grid = [[0] * 9 for _ in range(9)]

    # Fill in clue cells (using predictions/overrides)
    for (row, col), clue in game['clue_data'].items():
        if clue['overridden'] and clue['override_value'] is not None:
            user_grid[row][col] = clue['override_value']
        else:
            user_grid[row][col] = clue['predicted']

    # Fill in user inputs
    for (row, col), val in game['user_inputs'].items():
        if val is not None:
            user_grid[row][col] = val

    # Check for empty cells
    empty_cells = []
    for row in range(9):
        for col in range(9):
            if user_grid[row][col] == 0:
                empty_cells.append((row, col))

    if empty_cells:
        st.warning(f"⚠️ Please fill in all cells. {len(empty_cells)} cells are still empty.")
        return

    # Build predictions dict for hallucination checking
    predictions_for_check = {}
    for (row, col), clue in game['clue_data'].items():
        if not clue['overridden']:
            predictions_for_check[(row, col)] = clue['predicted']

    # Check solution
    is_correct, errors = check_solution(user_grid, game['solution'], predictions_for_check)

    if is_correct:
        st.success("🎉 Congratulations! You solved the puzzle correctly!")
        st.balloons()

        # Show stats
        hallucinations_caught = len([c for c in game['clue_data'].values()
                                     if c['overridden'] and c['predicted'] != c['ground_truth']])
        total_hallucinations = len(game['model_stats']['hallucinations'])

        if total_hallucinations > 0:
            st.info(f"🔍 You caught {hallucinations_caught}/{total_hallucinations} AI hallucinations!")
    else:
        st.error(f"❌ Not quite right. Found {len(errors)} error(s).")

        for error in errors:
            row, col = error['row'], error['col']
            user_val = error['user_value']
            correct_val = error['correct_value']

            if error['is_hallucination']:
                ai_pred = error['ai_predicted']
                st.markdown(
                    f"""<div class='hallucination-alert'>
                    <strong>🤖 Perception Error:</strong> The AI hallucinated a <strong>{ai_pred}</strong>
                    at row {row + 1}, column {col + 1}, but it was actually a <strong>{correct_val}</strong>.
                    </div>""",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""<div class='hallucination-alert'>
                    <strong>❌ Error:</strong> Cell at row {row + 1}, column {col + 1}
                    has <strong>{user_val}</strong> but should be <strong>{correct_val}</strong>.
                    </div>""",
                    unsafe_allow_html=True
                )


def main():
    """Main application entry point."""
    # Initialize game state if needed
    if 'game_state' not in st.session_state:
        with st.spinner("Loading model and generating puzzle..."):
            st.session_state.game_state = initialize_game()

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🧩 Perception Sudoku")
    st.markdown("""
    **Can you catch the AI hallucinations?**

    Normal Sudoku is a game of logic. Our game is a game of perception. 
    We don’t give the player the numbers. 
    We show the player photos of numbers, and an AI tries to read them for us, but AI makes mistakes.
    """)

    st.markdown("---")

    # Render the grid in a centered container
    grid_col1, grid_col2, grid_col3 = st.columns([1, 4, 1])
    with grid_col2:
        render_grid()

    st.markdown("---")

    # Check solution button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✅ Check Solution", type="primary"):
            check_user_solution()

    # Footer
    st.markdown("---")
    st.caption("Perception Sudoku - Catch the AI hallucinations!")


if __name__ == "__main__":
    main()
