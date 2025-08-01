import threading
import numpy as np
import time
import cv2
import queue
import pyrealsense2 as rs
from ultralytics import YOLO

class DualRealsenseCamera:
    def __init__(self, l515_serial="f1270669", d455_serial="151422253784", model_path="best.pt"):
        self.ctx = rs.context()
        self.l515_pipeline = rs.pipeline(self.ctx)
        self.d455_pipeline = rs.pipeline(self.ctx)
        self.l515_serial = l515_serial
        self.d455_serial = d455_serial
        
        # Camera configurations
        self.l515_config = rs.config()
        self.d455_config = rs.config()
        
        # Load YOLO model
        self.model = YOLO(model_path)
        
        # Camera state
        self.l515_frames = None
        self.d455_frames = None
        self.running = False
        self.l515_started = False
        self.d455_started = False
        self.frame_counter = 0
        self.detection_queue = queue.Queue(maxsize=1)
        
        # Camera intrinsics (will be set during startup)
        self.l515_camera_matrix = None
        self.l515_dist_coeffs = None
        self.d455_camera_matrix = None
        self.d455_dist_coeffs = None

    def start_cameras(self):
        try:
            print("\nStarting cameras...")
            
            # Try different configurations for L515
            l515_configs_to_try = [
                (640, 480, 30),  # Default
                (1280, 720, 30),
                (960, 540, 30),
                (640, 480, 15)
            ]
            
            l515_started = False
            for width, height, fps in l515_configs_to_try:
                if not l515_started:
                    print(f"\nTrying L515 config: {width}x{height}@{fps}fps")
                    self.l515_config = rs.config()
                    self.l515_config.enable_device(self.l515_serial)
                    self.l515_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                    self.l515_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                    
                    if self.l515_config.can_resolve(self.l515_pipeline):
                        print("Config works - starting pipeline...")
                        self.l515_pipeline.start(self.l515_config)
                        self.l515_started = True
                        l515_started = True
                        
                        # Get L515 intrinsics
                        l515_profile = self.l515_pipeline.get_active_profile()
                        l515_color_profile = rs.video_stream_profile(l515_profile.get_stream(rs.stream.color))
                        l515_intrinsics = l515_color_profile.get_intrinsics()
                        self.l515_camera_matrix = np.array([
                            [l515_intrinsics.fx, 0, l515_intrinsics.ppx],
                            [0, l515_intrinsics.fy, l515_intrinsics.ppy],
                            [0, 0, 1]
                        ])
                        self.l515_dist_coeffs = np.array(l515_intrinsics.coeffs[:4]).reshape((4, 1))
                        
                        if not self.wait_for_first_frame(self.l515_pipeline, "L515"):
                            self.l515_pipeline.stop()
                            self.l515_started = False
                            l515_started = False
                    else:
                        print("Config not supported")
            
            if not self.l515_started:
                print("\nWarning: Could not start L515 camera - continuing with D455 only")
            
            # D455 configuration
            print("\nStarting D455...")
            self.d455_config = rs.config()
            self.d455_config.enable_device(self.d455_serial)
            self.d455_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.d455_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            if self.d455_config.can_resolve(self.d455_pipeline):
                self.d455_pipeline.start(self.d455_config)
                self.d455_started = True
                
                # Get D455 intrinsics
                d455_profile = self.d455_pipeline.get_active_profile()
                d455_color_profile = rs.video_stream_profile(d455_profile.get_stream(rs.stream.color))
                d455_intrinsics = d455_color_profile.get_intrinsics()
                self.d455_camera_matrix = np.array([
                    [d455_intrinsics.fx, 0, d455_intrinsics.ppx],
                    [0, d455_intrinsics.fy, d455_intrinsics.ppy],
                    [0, 0, 1]
                ])
                self.d455_dist_coeffs = np.array(d455_intrinsics.coeffs[:4]).reshape((4, 1))
                
                if not self.wait_for_first_frame(self.d455_pipeline, "D455"):
                    self.d455_pipeline.stop()
                    self.d455_started = False
            else:
                print("D455 configuration not resolvable")
            
            if not self.d455_started and not self.l515_started:
                raise RuntimeError("Could not start either camera")
            
            # Create align objects
            self.l515_align = rs.align(rs.stream.color)
            self.d455_align = rs.align(rs.stream.color)
            
            self.running = True
            
            # Start capture threads for active cameras
            if self.l515_started:
                threading.Thread(target=self.l515_capture, daemon=True).start()
            if self.d455_started:
                threading.Thread(target=self.d455_capture, daemon=True).start()
            
            threading.Thread(target=self.detection_thread, daemon=True).start()
            
            print(f"\nCamera status - L515: {'Running' if self.l515_started else 'Not available'}, "
                  f"D455: {'Running' if self.d455_started else 'Not available'}")
            return True
            
        except Exception as e:
            print(f"\nStartup failed: {str(e)}")
            self.print_device_info()
            self.cleanup()
            return False

    def wait_for_first_frame(self, pipeline, name, timeout=5):
        """Wait for the first frame with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            frames = pipeline.try_wait_for_frames(100)
            if frames:
                print(f"{name} got first frame")
                return True
            print(".", end="", flush=True)
        print(f"\nTimeout waiting for {name} first frame")
        return False

    def l515_capture(self):
        while self.running and self.l515_started:
            try:
                frames = self.l515_pipeline.wait_for_frames()
                aligned_frames = self.l515_align.process(frames)
                self.l515_frames = aligned_frames
                self.frame_counter += 1
            except Exception as e:
                print(f"\nL515 capture error: {str(e)}")
                time.sleep(0.1)

    def d455_capture(self):
        while self.running and self.d455_started:
            try:
                frames = self.d455_pipeline.wait_for_frames()
                aligned_frames = self.d455_align.process(frames)
                self.d455_frames = aligned_frames
            except Exception as e:
                print(f"\nD455 capture error: {str(e)}")
                time.sleep(0.1)

    def get_images_and_depth(self):
        """Returns (l515_color, l515_depth, d455_color, d455_depth) or Nones if not available"""
        try:
            l515_color, l515_depth = None, None
            d455_color, d455_depth = None, None
            
            if self.l515_started and self.l515_frames:
                color_frame = self.l515_frames.get_color_frame()
                depth_frame = self.l515_frames.get_depth_frame()
                if color_frame:
                    l515_color = np.asanyarray(color_frame.get_data())
                if depth_frame:
                    l515_depth = np.asanyarray(depth_frame.get_data())
            
            if self.d455_started and self.d455_frames:
                color_frame = self.d455_frames.get_color_frame()
                depth_frame = self.d455_frames.get_depth_frame()
                if color_frame:
                    d455_color = np.asanyarray(color_frame.get_data())
                if depth_frame:
                    d455_depth = np.asanyarray(depth_frame.get_data())
            
            return l515_color, l515_depth, d455_color, d455_depth
            
        except Exception as e:
            print(f"\nImage processing error: {str(e)}")
            return None, None, None, None

    def detection_thread(self):
        """Continuous detection thread that updates the detection queue"""
        while self.running:
            try:
                l515_color, l515_depth, d455_color, d455_depth = self.get_images_and_depth()
                
                result = {
                    'timestamp': time.time(),
                    'detections': [],
                    'images': {
                        'l515_color': l515_color,
                        'l515_depth': l515_depth,
                        'd455_color': d455_color,
                        'd455_depth': d455_depth
                    }
                }

                # Process both camera images
                for camera_type, color_img, depth_img, camera_matrix in [
                    ('l515', l515_color, l515_depth, self.l515_camera_matrix),
                    ('d455', d455_color, d455_depth, self.d455_camera_matrix)
                ]:
                    if color_img is None:
                        continue

                    # Run YOLO detection
                    results = self.model(color_img, conf=0.1)
                    
                    for r in results:
                        boxes = r.boxes
                        masks = r.masks

                        for index in range(len(boxes)):
                            b = boxes[index].xyxy[0].to('cpu').detach().numpy().copy()
                            c = boxes[index].cls
                            conf = boxes[index].conf

                            center_x = int((b[0] + b[2]) / 2)
                            center_y = int((b[1] + b[3]) / 2)
                            
                            # Get depth if available
                            depth = None
                            if depth_img is not None:
                                depth = depth_img.get_distance(center_x, center_y)

                            # Get boundary points if mask available
                            boundary_points = None
                            boundary_depths = None
                            if masks and index < len(masks):
                                boundary_points = masks[index].xy[0]
                                if depth_img is not None:
                                    boundary_depths = [
                                        depth_img.get_distance(int(p[0]), int(p[1])) 
                                        for p in boundary_points
                                    ]

                            # Calculate 3D position if we have camera matrix and depth
                            position_3d = None
                            if camera_matrix is not None and depth is not None:
                                fx = camera_matrix[0, 0]
                                fy = camera_matrix[1, 1]
                                cx = camera_matrix[0, 2]
                                cy = camera_matrix[1, 2]
                                
                                x = (center_x - cx) * depth / fx
                                y = (center_y - cy) * depth / fy
                                z = depth
                                position_3d = (x, y, z)

                            detection = {
                                'class': self.model.names[int(c)],
                                'class_id': int(c),
                                'confidence': float(conf),
                                'bbox': b.tolist(),
                                'center': (center_x, center_y),
                                'depth': depth,
                                'position_3d': position_3d,
                                'boundary_points': boundary_points,
                                'boundary_depths': boundary_depths,
                                'camera': camera_type
                            }
                            result['detections'].append(detection)

                # Update detection queue
                if self.detection_queue.full():
                    self.detection_queue.get_nowait()
                self.detection_queue.put_nowait(result)

                # Display images (optional)
                for r in results:
                    annotated_img = r.plot()
                    cv2.imshow(f'YOLO Detection - {camera_type.upper()}', annotated_img)
                
                if l515_depth is not None:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(l515_depth, alpha=0.08), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('L515 Depth', depth_colormap)
                
                if d455_depth is not None:
                    depth_colormap = cv2.applyColorMap(
                        cv2.convertScaleAbs(d455_depth, alpha=0.08), 
                        cv2.COLORMAP_JET
                    )
                    cv2.imshow('D455 Depth', depth_colormap)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            except Exception as e:
                print(f"Detection error: {str(e)}")
                time.sleep(0.1)

    def get_latest_detection(self):
        """Get the latest detection result"""
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None

    def print_device_info(self):
        print("\nConnected RealSense devices:")
        devices = self.ctx.query_devices()
        for i, dev in enumerate(devices):
            print(f"{i+1}. {dev.get_info(rs.camera_info.name)}")
            print(f"   Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"   Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
            print(f"   USB: {dev.get_info(rs.camera_info.usb_type_descriptor)}")

    def cleanup(self):
        self.running = False
        time.sleep(0.5)  # Give threads time to stop
        if self.l515_started:
            self.l515_pipeline.stop()
        if self.d455_started:
            self.d455_pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera cleanup complete")


if __name__ == "__main__":
    # Initialize with your model path
    camera_system = DualRealsenseCamera(
        l515_serial="f1270669",
        d455_serial="151422253784",
        model_path='/home/yuezang/Desktop/d455/zyyolo8_1.pt'
    )
    
    if camera_system.start_cameras():
        try:
            while True:
                detection = camera_system.get_latest_detection()
                if detection:
                    print(f"\nDetection at {detection['timestamp']}:")
                    for det in detection['detections']:
                        print(f"  {det['class']} (ID: {det['class_id']}, Conf: {det['confidence']:.2f})")
                        print(f"    Camera: {det['camera']}")
                        print(f"    Position: {det['position_3d']}")
                        print(f"    Depth: {det['depth']:.2f}m")
                
                time.sleep(0.1)  # Prevent busy waiting
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            camera_system.cleanup()
    else:
        print("Failed to start cameras")