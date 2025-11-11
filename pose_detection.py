# Installation: pip install ultralytics opencv-python matplotlib pillow numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from ultralytics import YOLO
import torch
import math

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
    Download a sample video for testing
    """
    try:
        import requests
        # Sample video with various poses
        sample_url = "https://assets.mixkit.co/videos/preview/mixkit-group-of-people-walking-on-a-pedestrian-crossing-34556-large.mp4"
        video_path = "sample_poses.mp4"
        
        if os.path.exists(video_path):
            print("✅ Sample video already exists!")
            return video_path
            
        print("\n📥 Downloading sample video... (This may take a moment)")
        response = requests.get(sample_url, stream=True)
        
        with open(video_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
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
        return None

def calculate_angle(point1, point2, point3):
    """
    Calculate the angle between three points (in degrees)
    """
    try:
        # Convert to numpy arrays
        a = np.array(point1)
        b = np.array(point2) 
        c = np.array(point3)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)  # Avoid numerical errors
        angle = np.arccos(cosine_angle)
        
        return np.degrees(angle)
    except:
        return 0

def get_keypoint_coords(keypoints, index):
    """
    Get coordinates of a specific keypoint if confidence > threshold
    """
    if keypoints is None or len(keypoints) <= index:
        return None
    if keypoints[index][2] > 0.3:  # Confidence threshold
        return (keypoints[index][0], keypoints[index][1])
    return None

def analyze_pose_correctly(keypoints):
    """
    CORRECT pose analysis using body angles and proportions
    """
    if keypoints is None or len(keypoints) < 6:
        return "Unknown"
    
    # COCO keypoint indices
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    try:
        # Get keypoint coordinates
        left_shoulder = get_keypoint_coords(keypoints, LEFT_SHOULDER)
        right_shoulder = get_keypoint_coords(keypoints, RIGHT_SHOULDER)
        left_hip = get_keypoint_coords(keypoints, LEFT_HIP)
        right_hip = get_keypoint_coords(keypoints, RIGHT_HIP)
        left_knee = get_keypoint_coords(keypoints, LEFT_KNEE)
        right_knee = get_keypoint_coords(keypoints, RIGHT_KNEE)
        left_ankle = get_keypoint_coords(keypoints, LEFT_ANKLE)
        right_ankle = get_keypoint_coords(keypoints, RIGHT_ANKLE)
        
        # Calculate average positions
        shoulders = []
        hips = []
        knees = []
        ankles = []
        
        if left_shoulder is not None: shoulders.append(left_shoulder)
        if right_shoulder is not None: shoulders.append(right_shoulder)
        if left_hip is not None: hips.append(left_hip)
        if right_hip is not None: hips.append(right_hip)
        if left_knee is not None: knees.append(left_knee)
        if right_knee is not None: knees.append(right_knee)
        if left_ankle is not None: ankles.append(left_ankle)
        if right_ankle is not None: ankles.append(right_ankle)
        
        if not shoulders or not hips:
            return "Unknown"
        
        # Calculate average positions
        avg_shoulder = np.mean(shoulders, axis=0) if shoulders else None
        avg_hip = np.mean(hips, axis=0) if hips else None
        avg_knee = np.mean(knees, axis=0) if knees else None
        avg_ankle = np.mean(ankles, axis=0) if ankles else None
        
        # POSE CLASSIFICATION LOGIC
        
        # 1. Check for SITTING pose - using knee angles
        sitting_detected = False
        if left_hip is not None and left_knee is not None and left_ankle is not None:
            left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
            if left_knee_angle < 140:  # Bent knee indicates sitting
                sitting_detected = True
                
        if right_hip is not None and right_knee is not None and right_ankle is not None:
            right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
            if right_knee_angle < 140:  # Bent knee indicates sitting
                sitting_detected = True
        
        if sitting_detected:
            return "Sitting"
        
        # 2. Check for WALKING pose - leg angle differences
        walking_detected = False
        if (left_hip is not None and left_knee is not None and left_ankle is not None and
            right_hip is not None and right_knee is not None and right_ankle is not None):
            
            left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
            right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
            
            # Walking: significant difference between leg angles
            leg_angle_diff = abs(left_leg_angle - right_leg_angle)
            if leg_angle_diff > 30:  # One leg straight, one bent
                walking_detected = True
        
        if walking_detected:
            return "Walking"
        
        # 3. Check for STANDING pose - body alignment
        if avg_shoulder is not None and avg_hip is not None:
            vertical_diff = avg_hip[1] - avg_shoulder[1]
            
            # If we have ankle information, use body proportions
            if avg_ankle is not None:
                body_height = avg_ankle[1] - avg_shoulder[1]
                if body_height > 0:
                    shoulder_hip_ratio = vertical_diff / body_height
                    # Standing: reasonable vertical alignment
                    if 0.15 < shoulder_hip_ratio < 0.35:
                        return "Standing"
            
            # Fallback: use absolute vertical difference
            if 20 < vertical_diff < 150:  # Reasonable range for standing
                return "Standing"
        
        return "Unknown"
        
    except Exception as e:
        print(f"Pose analysis error: {e}")
        return "Unknown"

def draw_skeleton(image, keypoints, color):
    """
    Draw skeleton lines between keypoints
    """
    # COCO skeleton connections
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    for connection in skeleton:
        start_idx, end_idx = connection
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3):
            
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            
            cv2.line(image, start_point, end_point, color, 2)

def process_video_poses(model, video_path):
    """
    Process video and detect human poses in each frame - CORRECTED VERSION
    """
    print(f"\n🎥 Processing video: {video_path}")
    print("🕺 Starting ACCURATE HUMAN POSE DETECTION...")

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
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(
        *'mp4v')
    output_path = "act_out2.mp4"  # yoloimg/vid20.mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_humans = 0
    all_confidences = []
    pose_statistics = {"Standing": 0, "Sitting": 0, "Walking": 0, "Unknown": 0}
    start_time = time.time()
    
    print("\n🎯 Processing frames with ACCURATE pose detection...")
    print("💡 Using angle-based pose classification...")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Run YOLO pose detection
            results = model(frame, verbose=False, conf=0.5)
            
            # Process results
            human_count = 0
            frame_confidences = []
            
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                human_count = len(results[0].keypoints)
                total_humans += human_count
                
                # Process each detected human
                for i, keypoints in enumerate(results[0].keypoints):
                    if keypoints is None:
                        continue
                        
                    kp_data = keypoints.data[0].cpu().numpy()
                    
                    if len(kp_data) > 0:
                        # Filter confident keypoints
                        valid_kps = kp_data[kp_data[:, 2] > 0.3]
                        if len(valid_kps) > 3:  # Need at least 4 keypoints for good analysis
                            # Calculate bounding box from keypoints
                            x_coords = valid_kps[:, 0]
                            y_coords = valid_kps[:, 1]
                            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                            
                            # Expand bounding box
                            padding = 15
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(frame.shape[1], x2 + padding)
                            y2 = min(frame.shape[0], y2 + padding)
                            
                            # Calculate average confidence
                            avg_confidence = np.mean(valid_kps[:, 2])
                            frame_confidences.append(avg_confidence)
                            
                            # Analyze pose using CORRECT method
                            pose = analyze_pose_correctly(kp_data)
                            pose_statistics[pose] += 1
                            
                            # Color coding
                            if pose == "Standing":
                                color = (0, 255, 0)  # Green
                                text_color = (0, 200, 0)
                            elif pose == "Sitting":
                                color = (0, 165, 255)  # Orange
                                text_color = (0, 120, 255)
                            elif pose == "Walking":
                                color = (255, 0, 0)  # Blue
                                text_color = (200, 0, 0)
                            else:
                                color = (128, 128, 128)  # Gray
                                text_color = (100, 100, 100)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                            
                            # Draw pose label
                            pose_label = f"{pose}"
                            cv2.putText(frame, pose_label, (x1, y1-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
                            
                            # Draw confidence
                            conf_label = f"Conf: {avg_confidence:.2f}"
                            cv2.putText(frame, conf_label, (x1, y1-40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                            
                            # Draw keypoints and skeleton
                            draw_skeleton(frame, kp_data, color)
            
            all_confidences.extend(frame_confidences)
            
            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Humans: {human_count}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Show real-time pose statistics
            y_offset = 110
            cv2.putText(frame, "Live Pose Stats:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 35
            
            for pose in ["Standing", "Walking", "Sitting", "Unknown"]:
                count = pose_statistics[pose]
                cv2.putText(frame, f"{pose}: {count}", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            cv2.putText(frame, "Press 'q' to STOP", (10, y_offset + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Write frame to output video
            out.write(frame)
            
            # Display the frame
            cv2.imshow('Accurate Pose Detection - Press "q" to Stop', frame)
            cv2.setWindowProperty('Accurate Pose Detection - Press "q" to Stop', cv2.WND_PROP_TOPMOST, 1)
            
            frame_count += 1
            
            # Show progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed
                remaining_frames = total_frames - frame_count
                eta = remaining_frames / fps_processed if fps_processed > 0 else 0
                
                print(f"📊 {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) | "
                      f"Poses: {pose_statistics} | ETA: {eta:.1f}s")
            
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
    avg_humans_per_frame = total_humans / frame_count if frame_count > 0 else 0
    actual_fps = frame_count / processing_time if processing_time > 0 else 0
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    print(f"✅ ACCURATE Pose detection completed!")
    print(f"\n📊 Performance Summary:")
    print(f"   • Frames processed: {frame_count}/{total_frames}")
    print(f"   • Total processing time: {processing_time:.2f} seconds")
    print(f"   • Processing speed: {actual_fps:.1f} FPS")
    print(f"   • Total humans detected: {total_humans}")
    print(f"   • Average humans per frame: {avg_humans_per_frame:.2f}")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    
    print(f"\n🎯 FINAL Pose Statistics:")
    total_poses = sum(pose_statistics.values())
    for pose, count in pose_statistics.items():
        percentage = (count / total_poses * 100) if total_poses > 0 else 0
        print(f"   • {pose}: {count} ({percentage:.1f}%)")
    
    print(f"   • Output saved as: {output_path}")
    
    return output_path, processing_time, total_humans, frame_count, avg_confidence

def live_camera_pose_detection(model):
    """
    Real-time accurate human pose detection using camera feed
    """
    print("\n🎥 Starting REAL-TIME ACCURATE Pose Detection...")
    print("📋 Instructions:")
    print("   • Press 'q' to QUIT and save video")
    print("   • Press 's' to START/STOP recording")
    print("   • Accurate pose detection will be shown in real-time")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access camera.")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Video writer setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20.0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = "camera_accurate_poses.mp4"
    out = None
    is_recording = False
    
    print("✅ Camera started.")
    print(f"💾 Video will be saved as: {output_path}")
    
    frame_count = 0
    total_humans = 0
    all_confidences = []
    pose_statistics = {"Standing": 0, "Sitting": 0, "Walking": 0, "Unknown": 0}
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            
            # Run pose detection
            results = model(frame, verbose=False, conf=0.5)
            
            human_count = 0
            frame_confidences = []
            
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                human_count = len(results[0].keypoints)
                total_humans += human_count
                
                for i, keypoints in enumerate(results[0].keypoints):
                    if keypoints is None:
                        continue
                        
                    kp_data = keypoints.data[0].cpu().numpy()
                    
                    if len(kp_data) > 0:
                        valid_kps = kp_data[kp_data[:, 2] > 0.3]
                        if len(valid_kps) > 3:
                            x_coords = valid_kps[:, 0]
                            y_coords = valid_kps[:, 1]
                            x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                            x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                            
                            padding = 15
                            x1 = max(0, x1 - padding)
                            y1 = max(0, y1 - padding)
                            x2 = min(frame.shape[1], x2 + padding)
                            y2 = min(frame.shape[0], y2 + padding)
                            
                            avg_confidence = np.mean(valid_kps[:, 2])
                            frame_confidences.append(avg_confidence)
                            
                            pose = analyze_pose_correctly(kp_data)
                            pose_statistics[pose] += 1
                            
                            if pose == "Standing":
                                color = (0, 255, 0)
                            elif pose == "Sitting":
                                color = (0, 165, 255)
                            elif pose == "Walking":
                                color = (255, 0, 0)
                            else:
                                color = (128, 128, 128)
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(display_frame, pose, (x1, y1-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            
                            # Draw skeleton
                            draw_skeleton(display_frame, kp_data, color)
            
            all_confidences.extend(frame_confidences)
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info to display
            cv2.putText(display_frame, f"Humans: {human_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Show pose stats
            y_offset = 110
            for pose in ["Standing", "Walking", "Sitting", "Unknown"]:
                count = pose_statistics[pose]
                cv2.putText(display_frame, f"{pose}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 25
            
            # Recording status
            status = "🔴 RECORDING" if is_recording else "⚪ READY"
            cv2.putText(display_frame, status, (10, y_offset + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if is_recording else (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 's' to record, 'q' to quit", (10, y_offset + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Save frame if recording
            if is_recording and out is not None:
                save_frame = frame.copy()
                if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                    for i, keypoints in enumerate(results[0].keypoints):
                        if keypoints is None:
                            continue
                        kp_data = keypoints.data[0].cpu().numpy()
                        if len(kp_data) > 0:
                            valid_kps = kp_data[kp_data[:, 2] > 0.3]
                            if len(valid_kps) > 3:
                                x_coords = valid_kps[:, 0]
                                y_coords = valid_kps[:, 1]
                                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                                
                                padding = 15
                                x1 = max(0, x1 - padding)
                                y1 = max(0, y1 - padding)
                                x2 = min(frame.shape[1], x2 + padding)
                                y2 = min(frame.shape[0], y2 + padding)
                                
                                pose = analyze_pose_correctly(kp_data)
                                
                                if pose == "Standing":
                                    color = (0, 255, 0)
                                elif pose == "Sitting":
                                    color = (0, 165, 255)
                                elif pose == "Walking":
                                    color = (255, 0, 0)
                                else:
                                    color = (128, 128, 128)
                                
                                cv2.rectangle(save_frame, (x1, y1), (x2, y2), color, 3)
                                cv2.putText(save_frame, pose, (x1, y1-15), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.putText(save_frame, f"Humans: {human_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                out.write(save_frame)
            
            # Display
            cv2.imshow('Accurate Real-Time Pose Detection', display_frame)
            cv2.setWindowProperty('Accurate Real-Time Pose Detection', cv2.WND_PROP_TOPMOST, 1)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                is_recording = not is_recording
                if is_recording:
                    if out is None:
                        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    print("🔴 Started recording...")
                else:
                    print("⏹️ Stopped recording")
                    if out is not None:
                        out.release()
                        out = None
            
            elif key == ord('q'):
                print("⏹️ Quitting...")
                break
                
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        if out is not None:
            out.release()
        cap.release()
        cv2.destroyAllWindows()
    
    # Print summary
    processing_time = time.time() - start_time
    avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    print(f"\n📊 Real-time Pose Detection Summary:")
    print(f"   • Frames processed: {frame_count}")
    print(f"   • Total time: {processing_time:.2f} seconds")
    print(f"   • Average FPS: {frame_count/processing_time:.1f}")
    print(f"   • Total humans: {total_humans}")
    print(f"   • Average confidence: {avg_confidence:.3f}")
    
    print(f"\n🎯 Pose Statistics:")
    total_poses = sum(pose_statistics.values())
    for pose, count in pose_statistics.items():
        percentage = (count / total_poses * 100) if total_poses > 0 else 0
        print(f"   • {pose}: {count} ({percentage:.1f}%)")
    
    print(f"   • Video saved as: {output_path}")

# Main execution function
def main():
    """
    ACCURATE Human Pose Detector
    """
    print("🚀 ACCURATE HUMAN POSE DETECTOR")
    print("=" * 50)
    print("Using ANGLE-BASED pose classification for accurate results")
    print("Standing, Sitting, Walking detection with proper logic")
    print("=" * 50)
    
    # Load pose model
    try:
        model = YOLO('yolov8s-pose.pt')  # More accurate than nano
        print("✅ Loaded yolov8s-pose.pt model")
    except:
        try:
            model = YOLO('yolov8n-pose.pt')
            print("✅ Loaded yolov8n-pose.pt model")
        except Exception as e:
            print(f"❌ Could not load pose model: {e}")
            return
    
    # Get video source
    video_source = select_video()
    
    if video_source is None:
        print("❌ No valid video source selected.")
        return
    
    if video_source == "camera":
        live_camera_pose_detection(model)
    else:
        result = process_video_poses(model, video_source)
        
        if result[0] is not None:
            output_path, processing_time, total_humans, frame_count, avg_confidence = result
            print(f"\n✅ ACCURATE pose detection completed!")
            print(f"🎯 Output saved as: {output_path}")
        else:
            print("❌ Video processing failed.")

# Run the main function
if __name__ == "__main__":
    main()