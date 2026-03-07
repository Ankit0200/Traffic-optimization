# Real-Time Aerial Vehicle Turn Prediction

This project uses YOLO and a custom trained LSTM model to track vehicles in aerial highway video footage, extract their coordinate trajectories, and predict their turn intentions or exit zones in real-time.

## Project Structure

- `src/`: Python scripts for pipeline execution
  - `nine_zone_prob.py`: Generates the coordinate-based transitions and trajectories
  - `lstm_predictor.py`: Trains the LSTM sequence model on generated trajectories
  - `test_lstm_realtime.py`: Real-time YOLO tracking paired with live LSTM predictions
  - `evaluate_lstm.py`: Headless evaluation logic for measuring early-prediction accuracy
- `notebooks/`: Jupyter notebooks (e.g. `kaggle_upload.ipynb`)
- `models/`: Where trained YOLO (`.pt`) and PyTorch LSTM models are stored
- `data/`: Trajectories, JSON configurations, log files, and videos
- `misc/`: Testing and utility scripts

## Usage

1. **Extract Trajectories**
   ```bash
   cd src
   python3 nine_zone_prob.py --video ../data/videos/your_video.mp4
   ```

2. **Train LSTM Prediction Model**
   ```bash
   python3 lstm_predictor.py --data your_video_trajectories.json
   ```

3. **Run Real-Time Prediction**
   ```bash
   python3 test_lstm_realtime.py --video ../data/videos/your_video.mp4 --model your_video_trajectories_lstm_model.pt
   ```

## Dependencies
- `torch`
- `ultralytics` (YOLOv8)
- `opencv-python`
- `numpy`
- `scikit-learn`
