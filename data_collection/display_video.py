# mypy: ignore-errors
import h5py
import cv2
import numpy as np
import argparse

def display_video(file_path, camera_name, playback_speed=1.0):
    with h5py.File(file_path, 'r') as f:
        images = f[f'observations/images/{camera_name}']
        num_frames, height, width, channels = images.shape
        print(images.shape)
        
        for i in range(num_frames):
            frame = images[i]
            
            # Convert from RGB to BGR (OpenCV uses BGR)
            #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow(f'Video Playback - {camera_name}', frame)
            
            # Calculate delay to achieve desired playback speed
            delay = int(1000 / (30 * playback_speed))  # Assuming 30 fps
            
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display video from HDF5 file')
    parser.add_argument('file_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('camera_name', type=int, help='Name of the camera (e.g., cam_high, cam_low, cam_left_wrist, cam_right_wrist)')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (default: 1.0)')
    
    args = parser.parse_args()
    
    display_video(args.file_path, args.camera_name, args.speed)
