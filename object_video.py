# Installation: pip install ultralytics opencv-python matplotlib pillow numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from ultralytics import YOLO
import torch

def select_video():
    """
    Handle video input for VS Code
    """
    print("📁 Video Selection Options:")
    print("1. Enter full path to MP4 video")
    print("2. Download sample video")
    print("3. Live Camera Feed (Real-time human detection)")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Option 1: User enters full path
        video_path = input("\nEnter the full path to your MP4 video: ").strip().strip('"')
        
        # Check if video exists and is valid
        if video_path and os.path.exists(video_path):
            if is_valid_video(video_path):
                print(f"✅ Video found and valid: {video_path}")
                return video_path
            else:
                print("❌ Video file is corrupted or invalid format")
                return None
        else:
            print("❌ Video not found. Please check the path.")
            return None
            
    elif choice == "2":
        # Option 2: Download sample video
        video_path = download_sample_video()
        return video_path
        
    elif choice == "3":
        # Option 3: Live camera feed for real-time detection
        return "camera"
        
    else:
        print("❌ Invalid option selected.")
        return None

def is_valid_video(video_path):
    """
    Check if the video file is valid and can be opened
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except:
        return False

def download_sample_video():
    """
    Download a sample video for testing - using a more reliable source
    """
    try:
        import requests
        # Using a smaller, more reliable sample video
        # sample_url = "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
        sample_url = "https://www.pexels.com/download/video/3254013/"
        video_path = "sample_vid.mp4"
        
        if os.path.exists(video_path):
            print("✅ Sample video already exists!")
            return video_path
            
        print("📥 Downloading sample video... (This may take a moment)")
        response = requests.get(sample_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(video_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"📥 Download progress: {progress:.1f}%", end='\r')
        
        print("\n✅ Sample video downloaded!")
        
        # Verify the downloaded video
        if is_valid_video(video_path):
            return video_path
        else:
            print("❌ Downloaded video is corrupted")
            return None
            
    except ImportError:
        print("❌ Requests module not available. Please install: pip install requests")
        return None
    except Exception as e:
        print(f"❌ Failed to download sample video: {e}")
        # Fallback to creating a sample video from camera
        return create_sample_video()

def create_sample_video():
    """
    Create a simple sample video using camera
    """
    try:
        video_path = "camera_sample.mp4"
        print("🎬 Creating sample video from camera...")
        print("📹 Please allow camera access. Recording 5 seconds...")
        
        # Create a simple video with OpenCV
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera to create sample video")
            return None
            
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20.0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Record 5 seconds of video
        start_time = time.time()
        recorded_frames = 0
        
        while (time.time() - start_time) < 5:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                recorded_frames += 1
                
                # Show preview
                cv2.imshow('Creating Sample Video...', frame)
                cv2.waitKey(1)
            else:
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        if recorded_frames > 0:
            print(f"✅ Sample video created with {recorded_frames} frames!")
            return video_path
        else:
            print("❌ Failed to create sample video - no frames recorded")
            return None
        
    except Exception as e:
        print(f"❌ Failed to create sample video: {e}")
        return None

def process_video_humans(model, video_path):
    """
    Process video and detect ALL objects in each frame
    """
    print(f"\n🎥 Processing video: {video_path}")
    print("⏱️ Starting OBJECT DETECTION...")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return None, None, None, None, None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if fps <= 0 or total_frames <= 0:
        print("❌ Error: Invalid video properties")
        cap.release()
        return None, None, None, None, None
    
    print(f"📊 Video Info: {width}x{height}, {fps:.1f} FPS, {total_frames} frames")
    print(f"⏳ Estimated processing time: {total_frames/fps/3:.1f} seconds (approx)")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "outs.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_objects = 0
    all_confidences = []
    start_time = time.time()
    
    print("\n🎯 Processing frames... (Press 'q' to stop early)")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # ⭐ CHANGED: Run YOLO detection on ALL objects (removed classes=[0])
            results = model(frame, verbose=False)
            
            # Process results
            object_count = 0
            frame_confidences = []
            
            if results[0].boxes is not None:
                object_count = len(results[0].boxes)
                total_objects += object_count
                
                # Draw bounding boxes on the frame for ALL objects
                for box in results[0].boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]  # ⭐ ADDED: Get object name
                    
                    frame_confidences.append(confidence)
                    
                    # Draw rectangle with different colors for different classes
                    color = (0, 255, 0)  # Default green
                    if class_name == 'person':
                        color = (0, 255, 0)  # Green for humans
                    elif class_name == 'car':
                        color = (255, 0, 0)  # Blue for cars
                    else:
                        color = (0, 165, 255)  # Orange for other objects
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # ⭐ CHANGED: Draw label with object name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            all_confidences.extend(frame_confidences)
            
            # ⭐ CHANGED: Update display text
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Objects: {object_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if frame_confidences:
                avg_frame_confidence = np.mean(frame_confidences)
                cv2.putText(frame, f"Avg Conf: {avg_frame_confidence:.3f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, "Press 'q' to STOP", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Video Object Detection - Press "q" to Stop', frame)
            cv2.setWindowProperty('Video Object Detection - Press "q" to Stop', cv2.WND_PROP_TOPMOST, 1)
            
            frame_count += 1
            
            # Show progress every 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed
                remaining_frames = total_frames - frame_count
                eta = remaining_frames / fps_processed if fps_processed > 0 else 0
                
                current_avg_confidence = np.mean(all_confidences) if all_confidences else 0
                print(f"📊 Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%) - ETA: {eta:.1f}s - Avg Conf: {current_avg_confidence:.3f}")
            
            # Check for 'q' key to stop early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("⏹️ Video processing stopped by user")
                break
                
    except Exception as e:
        print(f"❌ Error during video processing: {e}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return None, None, None, None, None
    
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Calculate performance metrics
    processing_time = time.time() - start_time
    avg_objects_per_frame = total_objects / frame_count if frame_count > 0 else 0
    actual_fps = frame_count / processing_time if processing_time > 0 else 0
    
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    confidence_std = np.std(all_confidences) if all_confidences else 0
    
    print(f"✅ Video processing completed!")
    print(f"\n📊 Performance Summary:")
    print(f"   • Frames processed: {frame_count}/{total_frames}")
    print(f"   • Total processing time: {processing_time:.2f} seconds")
    print(f"   • Processing speed: {actual_fps:.1f} FPS")
    print(f"   • ⭐ Total objects detected: {total_objects}")  # ⭐ CHANGED
    print(f"   • ⭐ Average objects per frame: {avg_objects_per_frame:.2f}")  # ⭐ CHANGED
    print(f"   • ⭐ Average confidence: {avg_confidence:.3f}")
    print(f"   • ⭐ Confidence std dev: {confidence_std:.3f}")
    print(f"   • Output saved as: {output_path}")
    
    return output_path, processing_time, total_objects, frame_count, avg_confidence

def live_camera_detection(model):
    """
    Real-time ALL object detection using camera feed
    """
    print("\n🎥 Starting REAL-TIME Object Detection...")
    print("📋 Instructions:")
    print("   • Press 'q' to QUIT")
    print("   • ALL object detections will be shown in real-time")
    print("   • Make sure your camera is connected")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access camera. Please check if camera is connected.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Camera started. Press 'q' to exit real-time detection.")
    
    frame_count = 0
    total_objects = 0
    all_confidences = []
    start_time = time.time()
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # ⭐ CHANGED: Run YOLO detection on ALL objects (removed classes=[0])
            results = model(frame, verbose=False)
            
            # Process results
            object_count = 0
            frame_confidences = []
            
            if results[0].boxes is not None:
                object_count = len(results[0].boxes)
                total_objects += object_count
                
                # Draw bounding boxes on the frame for ALL objects
                for box in results[0].boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]  # ⭐ ADDED: Get object name
                    
                    frame_confidences.append(confidence)
                    
                    # Draw rectangle with different colors for different classes
                    color = (0, 255, 0)  # Default green
                    if class_name == 'person':
                        color = (0, 255, 0)  # Green for humans
                    elif class_name == 'car':
                        color = (255, 0, 0)  # Blue for cars
                    else:
                        color = (0, 165, 255)  # Orange for other objects
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # ⭐ CHANGED: Draw label with object name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            all_confidences.extend(frame_confidences)
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Calculate current average confidence
            current_avg_confidence = np.mean(all_confidences) if all_confidences else 0
            
            # ⭐ CHANGED: Update display text
            cv2.putText(frame, f"Objects: {object_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Avg Conf: {current_avg_confidence:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to QUIT", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Real-Time Object Detection - Press "q" to Quit', frame)
            cv2.setWindowProperty('Real-Time Object Detection - Press "q" to Quit', cv2.WND_PROP_TOPMOST, 1)
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("✅ Real-time detection stopped.")
                break
                
    except Exception as e:
        print(f"❌ Error during real-time detection: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    processing_time = time.time() - start_time
    avg_objects_per_frame = total_objects / frame_count if frame_count > 0 else 0
    
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    confidence_std = np.std(all_confidences) if all_confidences else 0
    
    print(f"\n📊 Real-time Detection Summary:")
    print(f"   • Frames processed: {frame_count}")
    print(f"   • Total processing time: {processing_time:.2f} seconds")
    print(f"   • Average FPS: {frame_count/processing_time:.1f}")
    print(f"   • ⭐ Total objects detected: {total_objects}")  # ⭐ CHANGED
    print(f"   • ⭐ Average objects per frame: {avg_objects_per_frame:.2f}")  # ⭐ CHANGED
    print(f"   • ⭐ Average confidence: {avg_confidence:.3f}")
    print(f"   • ⭐ Confidence std dev: {confidence_std:.3f}")

# Main execution function
def main():
    """
    YOLO Human Detector for VS Code - VIDEO PROCESSING
    """
    print("🚀 YOLO HUMAN DETECTOR - VIDEO PROCESSING")
    print("=" * 50)
    print("This program will detect humans in videos/camera feed")
    print("All other objects will be ignored")
    print("=" * 50)
    
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Get video path or camera option
    video_source = select_video()
    
    if video_source is None:
        print("❌ No valid video source selected. Exiting...")
        return
    
    if video_source == "camera":
        # Real-time camera detection
        live_camera_detection(model)
    else:
        # Process video file
        result = process_video_humans(model, video_source)
        
        if result[0] is not None:  # Check if processing was successful
            output_path, processing_time, total_humans, frame_count, avg_confidence = result
            print(f"\n✅ Video human detection completed successfully!")
            print(f"🎯 Output video saved as: {output_path}")
            print(f"⭐ Final Average Confidence: {avg_confidence:.3f}")  # ⭐ NEW
        else:
            print("❌ Video processing failed. Please check your video file.")

# Run the main function
if __name__ == "__main__":
    main()
    # yoloimg/vid1.mp4