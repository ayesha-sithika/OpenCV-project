# Installation: pip install ultralytics opencv-python matplotlib pillow numpy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import time
from ultralytics import YOLO
import torch

def select_image():
    """
    Handle image input for VS Code
    """
    print("📁 Image Selection Options:")
    print("1. Enter full path to image")
    print("2. Download sample image")
    print("3. Live Camera Feed (Capture image with 'c')")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Option 1: User enters full path
        image_path = input("\nEnter the full path to your image: ").strip().strip('"')
        
        # Check if image exists
        if image_path and os.path.exists(image_path):
            print(f"✅ Image found: {image_path}")
            return image_path
        else:
            print("❌ Image not found. Please check the path.")
            return None
            
    elif choice == "2":
        # Option 2: Download sample
        image_path = download_sample_image()
        return image_path
        
    elif choice == "3":
        # Option 3: Live camera feed
        image_path = capture_from_camera()
        return image_path
        
    else:
        print("❌ Invalid option selected.")
        return None

def download_sample_image():
    """
    Download a sample image
    """
    try:
        import requests
        sample_url = "https://ultralytics.com/images/bus.jpg"
        image_path = "sample_image.jpg"
        
        print("📥 Downloading sample image...")
        response = requests.get(sample_url)
        with open(image_path, 'wb') as f:
            f.write(response.content)
        
        print("✅ Sample image downloaded!")
        return image_path
    except ImportError:
        print("❌ Requests module not available. Please install: pip install requests")
        return None
    except Exception as e:
        print(f"❌ Failed to download sample image: {e}")
        return None

def capture_from_camera():
    """
    Capture image from live camera feed
    """
    print("\n📷 Starting camera feed...")
    print("📋 Instructions:")
    print("   • Press 'c' to CAPTURE image")
    print("   • Press 'q' to QUIT camera")
    print("   • Make sure your camera is connected and accessible")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not access camera. Please check if camera is connected.")
        return None
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    captured_image_path = None
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add instructions to the frame
            cv2.putText(frame, "Press 'c' to CAPTURE", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to QUIT", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Camera Feed - Press "c" to Capture, "q" to Quit', frame)
            cv2.setWindowProperty('Camera Feed - Press "c" to Capture, "q" to Quit', cv2.WND_PROP_TOPMOST, 1)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                # Capture image
                captured_image_path = "captured_image.jpg"
                cv2.imwrite(captured_image_path, frame)
                print(f"✅ Image captured and saved as: {captured_image_path}")
                break
                
            elif key == ord('q'):
                print("❌ Camera feed closed without capture.")
                break
                
    except Exception as e:
        print(f"❌ Error during camera capture: {e}")
    
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
    
    return captured_image_path

def filter_human_detections(results):
    """
    Filter results to only include human detections (class_id = 0)
    Simplified version that modifies the original results in-place
    """
    for r in results:
        if r.boxes is not None:
            # Get indices of human detections (class 0)
            human_indices = []
            for i, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                if class_id == 0:  # Only keep humans
                    human_indices.append(i)
            
            if human_indices:
                # Filter boxes using the indices
                r.boxes = r.boxes[human_indices]
            else:
                r.boxes = None
    
    return results

def benchmark_detection(model, image_path):
    """
    Run detection and measure performance - HUMAN ONLY
    """
    print(f"\n⏱️ Starting HUMAN-ONLY detection on VS Code (CPU)...")
    
    # Start timer
    start_time = time.time()
    
    # Run YOLO detection ONLY for humans (class 0)
    results = model(image_path, classes=[0])
    
    # Calculate processing time
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"✅ Human detection completed in {processing_time:.2f} seconds")
    
    return results, processing_time

def display_results(model, results, image_path, processing_time):
    """
    Display detection results with performance info - ONLY HUMANS
    """
    for i, r in enumerate(results):
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        
        # Display detection details
        if r.boxes is not None and len(r.boxes) > 0:
            human_count = len(r.boxes)
            print(f"\n👥 Humans Detected: {human_count}")
            print("-" * 40)
            
            for j, box in enumerate(r.boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                if j < 10:  # Show first 10 detections
                    print(f"  {j+1}. {class_name:12s} | Confidence: {confidence:.3f}")
            
            if human_count > 10:
                print(f"  ... and {human_count - 10} more humans")
            
            print(f"\n📈 Summary:")
            print(f"  • Total humans detected: {human_count}")
            
            # Show confidence statistics
            confidences = [float(box.conf[0]) for box in r.boxes]
            avg_confidence = np.mean(confidences)
            print(f"  • Average confidence: {avg_confidence:.3f}")
            
        else:
            print("❌ No humans detected in the image.")
            human_count = 0
        
        # Create visualization
        print("\n🎨 Generating visualization...")
        
        # Plot the results (this will only show humans due to our class filtering)
        im_array = r.plot()
        im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        
        # Display comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Original image
        try:
            original_img = Image.open(image_path)
            ax1.imshow(original_img)
        except:
            # If PIL can't open, use OpenCV
            original_img = cv2.imread(image_path)
            if original_img is not None:
                original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
                ax1.imshow(original_img_rgb)
            else:
                print("❌ Could not load original image for display")
                return
        
        ax1.set_title('Original Image\nVS Code (CPU)', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Detected image (only humans)
        ax2.imshow(im_rgb)
        ax2.set_title(f'HUMAN DETECTION\nHumans: {human_count}', 
                     fontsize=14, fontweight='bold', color='red')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

def live_camera_detection(model):
    """
    Real-time human detection using camera feed
    """
    print("\n🎥 Starting REAL-TIME Human Detection...")
    print("📋 Instructions:")
    print("   • Press 'q' to QUIT")
    print("   • Human detections will be shown in real-time")
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
    
    try:
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Could not read frame from camera.")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Run YOLO detection on the frame (only humans)
            results = model(frame, classes=[0], verbose=False)
            
            # Process results
            human_count = 0
            if results[0].boxes is not None:
                human_count = len(results[0].boxes)
                
                # Draw bounding boxes on the frame
                for box in results[0].boxes:
                    # Get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"Human: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add human count and instructions to frame
            cv2.putText(frame, f"Humans: {human_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to QUIT", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Real-Time Human Detection - Press "q" to Quit', frame)
            
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

# Main execution function
def main():
    """
    YOLO Human Detector for VS Code - ONLY DETECTS HUMANS
    """
    # Load YOLO model
    model = YOLO('yolov8n.pt')
    
    # Get image path or camera option
    image_path = select_image()
    
    if image_path is None:
        return
    
    # Special case: if user wants real-time detection
    if image_path == "realtime":
        live_camera_detection(model)
        return
    
    # Choose detection method (only for static images)
    print("\n🎯 Detection Methods:")
    print("1. Human-only detection (recommended - fastest)")
    print("2. Manual filtering with confidence threshold")
    
    method_choice = input("Choose method (1/2, default 1): ").strip()
    
    if method_choice == "2":
        # Manual method with confidence threshold
        conf_threshold = float(input("Enter confidence threshold (0.1-0.9, default 0.25): ") or "0.25")
        
        # For manual method, run detection and display
        print(f"\n⏱️ Starting manual human detection...")
        start_time = time.time()
        results = model(image_path)
        human_detections = []
        
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if class_id == 0 and confidence >= conf_threshold:
                        human_detections.append({
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"⏱️ Processing Time: {processing_time:.2f} seconds")
        print(f"🔍 Confidence threshold: {conf_threshold}")
        print(f"👥 Humans detected: {len(human_detections)}")
        
        for i, detection in enumerate(human_detections[:10]):
            print(f"  {i+1}. Confidence: {detection['confidence']:.3f}")
        
        if len(human_detections) > 10:
            print(f"  ... and {len(human_detections) - 10} more humans")
        
        # Also show the visual results
        results, _ = benchmark_detection(model, image_path)
        display_results(model, results, image_path, processing_time)
            
    else:
        # Method 1: Direct human-only detection (recommended)
        results, processing_time = benchmark_detection(model, image_path)
        
        # Display results
        display_results(model, results, image_path, processing_time)
    
    print(f"\n✅ Human detection completed successfully!")

# Run the main function
if __name__ == "__main__":
    main()
    # yoloimg/sample1.jpg
    