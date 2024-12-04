import torch
from torchvision.transforms import functional as F
import cv2
from pathlib import Path
from rich.progress import track

def extract_frames(video_path, output_folder, desired_fps=6):
    video_path = Path(video_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(str(video_path))

    # Get the original video frame rate
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        print("Warning: Unable to get FPS of the video. Defaulting to 24 fps.")
        original_fps = 24  # Assume 24 fps if unable to get FPS

    # Calculate the interval between frames to capture
    frame_interval = int(original_fps / desired_fps)
    if frame_interval == 0:
        frame_interval = 1  # Ensure at least every frame is considered

    success, image = vidcap.read()
    frame_count = 0  # Total frames processed
    saved_count = 0  # Frames saved

    while success:
        if frame_count % frame_interval == 0:
            frame_path = output_folder / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_path), image)  # Save frame as JPEG file
            saved_count += 1
        success, image = vidcap.read()
        frame_count += 1

    vidcap.release()

def batch_extract_frames(video_folder, output_base_folder, desired_fps=6):
    video_folder = Path(video_folder)
    output_base_folder = Path(output_base_folder)
    videos = list(video_folder.glob('*.mp4'))
    for video_file in track(videos, total = len(videos), description="Extracting frames"):
        output_folder = output_base_folder / video_file.stem
        extract_frames(video_file, output_folder, desired_fps=desired_fps)

def batch_process_frames(model, base_frames_folder, output_base_folder):
    base_frames_folder = Path(base_frames_folder)
    output_base_folder = Path(output_base_folder)
    frames_subfolders = [x for x in base_frames_folder.iterdir() if x.is_dir()]
    for frames_folder in track(frames_subfolders, total = len(frames_subfolders), description="Processing frames"):
        output_folder = output_base_folder / frames_folder
        process_frames(model, frames_folder, output_folder)

def process_frames(model, frames_folder, output_folder, device=None):
    frames_folder = Path(frames_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    frame_files = sorted(frames_folder.iterdir())
    for frame_id, frame_file in enumerate(frame_files):
        if frame_file.is_file() and frame_file.suffix.lower() in ['.jpg', '.png']:
            image = cv2.imread(str(frame_file))
            predictions = get_hand_boxes(model, image, device)
            boxes = filter_predictions(predictions, threshold=0.5)
            crop_hands(image, boxes, output_folder, frame_id)

def crop_hands(image, boxes, output_folder, frame_id):
    for idx, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.astype(int)
        xmax = xmin + 256
        ymax = ymin + 256
        hand_crop = image[ymin:ymax, xmin:xmax]
        crop_path = output_folder / f"frame_{frame_id:05d}_hand_{idx}.jpg"
        cv2.imwrite(str(crop_path), hand_crop)

def get_hand_boxes(model, image, device):
    # Transform the image
    img = F.to_tensor(image)
    # Add batch dimension
    img = img.unsqueeze(0)
    # Move to device
    img = img.to(device)
    # Get predictions
    with torch.no_grad():
        predictions = model(img)
    return predictions[0]

def filter_predictions(predictions, threshold=0.5):
    boxes = predictions['boxes']
    scores = predictions['scores']
    selected_indices = scores > threshold
    selected_boxes = boxes[selected_indices]
    return selected_boxes.cpu().numpy()

def main():
    raw_video_path = '/data/val_rgb_front_clips/raw_videos/'
    frames_folder = '/data/val_rgb_front_clips/frames/'
    processed_frames_folder = '/data/val_rgb_front_clips/processed/'
    desired_fps = 6

    batch_extract_frames(raw_video_path, frames_folder, desired_fps)
    model = load_hand_detection_model()
    batch_process_frames(model, frames_folder, processed_frames_folder)

def load_hand_detection_model():
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

if __name__ == "__main__":
    main()

