import threading
import numpy as np
import time
import cv2
import queue
import pyrealsense2 as rs

from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R


class DualRealsenseCamera:
    def __init__(self, l515_serial="f1270669", d455_serial="151422253784"):
        self.ctx = rs.context()
        self.l515_pipeline = rs.pipeline(self.ctx)
        self.d455_pipeline = rs.pipeline(self.ctx)
        
        # Camera configurations
        self.l515_config = rs.config()
        self.d455_config = rs.config()
        
        self.l515_config.enable_device(l515_serial)
        self.d455_config.enable_device(d455_serial)
        
        # Conservative settings that should work
        self.l515_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        self.d455_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        
        # Camera matrices and distortion coefficients
        self.l515_camera_matrix = None
        self.l515_dist_coeffs = None
        self.d455_camera_matrix = None
        self.d455_dist_coeffs = None
        
        self.l515_frames = None
        self.d455_frames = None
        self.running = False
        self.l515_started = False
        self.d455_started = False
        self.frame_counter = 0
        self.detection_queue = queue.Queue(maxsize=1)

    def start_cameras(self):
        try:
            print("\nTesting camera configurations...")
            print("L515 can resolve:", self.l515_config.can_resolve(self.l515_pipeline))
            print("D455 can resolve:", self.d455_config.can_resolve(self.d455_pipeline))
            
            print("\nStarting L515...")
            self.l515_pipeline.start(self.l515_config)
            self.l515_started = True
            print("L515 started - waiting for first frame...")
            if not self.wait_for_first_frame(self.l515_pipeline, "L515"):
                return False
            
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
            
            print("\nStarting D455...")
            self.d455_pipeline.start(self.d455_config)
            self.d455_started = True
            print("D455 started - waiting for first frame...")
            if not self.wait_for_first_frame(self.d455_pipeline, "D455"):
                return False
            
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
            
            self.running = True
            threading.Thread(target=self.l515_capture, daemon=True).start()
            threading.Thread(target=self.d455_capture, daemon=True).start()
            threading.Thread(target=self.detection_thread, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"\nStartup failed: {str(e)}")
            self.print_device_info()
            self.cleanup()
            return False

    def wait_for_first_frame(self, pipeline, name):
        """Wait up to 5 seconds for the first frame"""
        for _ in range(50):
            frames = pipeline.try_wait_for_frames(100)
            if frames:
                print(f"{name} got first frame")
                return True
            print(".", end="", flush=True)
        print(f"\nTimeout waiting for {name} first frame")
        return False

    def l515_capture(self):
        while self.running:
            try:
                frames = self.l515_pipeline.wait_for_frames(1000)  # 1 second timeout
                self.l515_frames = frames
                self.frame_counter += 1
            except Exception as e:
                print(f"\nL515 capture error: {str(e)}")

    def d455_capture(self):
        while self.running:
            try:
                frames = self.d455_pipeline.wait_for_frames(1000)  # 1 second timeout
                self.d455_frames = frames
            except Exception as e:
                print(f"\nD455 capture error: {str(e)}")

    def get_images(self):
        """Returns (l515_image, d455_image) or (None, None) if frames not available"""
        try:
            l515_img = None
            d455_img = None
            
            if self.l515_frames:
                color_frame = self.l515_frames.get_color_frame()
                if color_frame:
                    l515_img = np.asanyarray(color_frame.get_data())
            
            if self.d455_frames:
                color_frame = self.d455_frames.get_color_frame()
                if color_frame:
                    d455_img = np.asanyarray(color_frame.get_data())
            
            return l515_img, d455_img
            
        except Exception as e:
            print(f"\nImage processing error: {str(e)}")
            return None, None

    def transform_pose_from_8_to_7(self, tvec, rvec):
        """Transform pose from ID8 to ID7 with corrected multi-axis rotation"""
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        # Create combined rotation correction:
        # 1. Original -90° Z rotation
        # 2. Additional X and Y corrections from analysis
        z_correction = R.from_euler('z', -90, degrees=True)
        xy_correction = R.from_euler('xy', [0, 0], degrees=True)
        
        # Combine rotations (order matters: Z first, then XY)
        total_correction = (xy_correction * z_correction).as_matrix()
        
        # Apply full correction
        new_rot = rot_matrix @ total_correction

        # Apply translation adjustment (-5mm X, +10mm Y, -5mm Z)
        new_tvec = tvec + rot_matrix @ np.array([[-0.005], [0.01], [-0.005]])
        
        # Convert back to rotation vector
        new_rvec, _ = cv2.Rodrigues(new_rot)
        return new_tvec, new_rvec

    def detection_thread(self):
        """Continuous detection thread that updates the detection queue"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        marker_size = 0.011  # 11mm in meters

        while self.running:
            try:
                l515_img, d455_img = self.get_images()
                if l515_img is None or d455_img is None:
                    time.sleep(0.1)
                    continue

                result = {
                    'timestamp': time.time(),
                    'object_detected': False,
                    'position': None,
                    'rotation': None,
                    'camera_source': None
                }

                # Process D455 image (IDs 7 and 8)
                d455_corners, d455_ids, _ = detector.detectMarkers(d455_img)
                if d455_ids is not None:
                    obj_points = np.array([
                        [-marker_size/2, marker_size/2, 0],
                        [marker_size/2, marker_size/2, 0],
                        [marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]
                    ], dtype=np.float32)

                    for i, id in enumerate(d455_ids.flatten()):
                        ret, rvec, tvec = cv2.solvePnP(
                            obj_points, d455_corners[i], 
                            self.d455_camera_matrix, self.d455_dist_coeffs
                        )
                        if ret and id == 8:
                            tvec7, rvec7 = self.transform_pose_from_8_to_7(tvec, rvec)
                            result.update({
                                'object_detected': True,
                                'position': tvec7.flatten(),
                                'rotation': rvec7.flatten(),
                                'camera_source': 'd455_id8'
                            })
                            break
                        elif ret and id == 7:
                            result.update({
                                'object_detected': True,
                                'position': tvec.flatten(),
                                'rotation': rvec.flatten(),
                                'camera_source': 'd455_id7'
                            })
                            break

                # Process L515 image (ID 9) only if D455 didn't detect anything
                if not result['object_detected']:
                    l515_corners, l515_ids, _ = detector.detectMarkers(l515_img)
                    if l515_ids is not None:
                        obj_points = np.array([
                            [-marker_size/2, marker_size/2, 0],
                            [marker_size/2, marker_size/2, 0],
                            [marker_size/2, -marker_size/2, 0],
                            [-marker_size/2, -marker_size/2, 0]
                        ], dtype=np.float32)

                        for i, id in enumerate(l515_ids.flatten()):
                            ret, rvec, tvec = cv2.solvePnP(
                                obj_points, l515_corners[i], 
                                self.l515_camera_matrix, self.l515_dist_coeffs
                            )
                            if ret and id == 8:
                                result.update({
                                    'object_detected': True,
                                    'position': tvec.flatten(),
                                    'rotation': rvec.flatten(),
                                    'camera_source': 'l515_id9'
                                })
                                break

                # Update detection queue
                if self.detection_queue.full():
                    self.detection_queue.get_nowait()
                self.detection_queue.put_nowait(result)

                # Display images (optional)
                if d455_corners is not None:
                    cv2.aruco.drawDetectedMarkers(d455_img, d455_corners, d455_ids)
                if l515_corners is not None:
                    cv2.aruco.drawDetectedMarkers(l515_img, l515_corners, l515_ids)
                
                cv2.imshow('L515 (ID9 detection)', l515_img)
                cv2.imshow('D455 (ID7/8 detection)', d455_img)
                cv2.waitKey(1)

            except Exception as e:
                print(f"Detection error: {str(e)}")
                continue

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
        time.sleep(0.5)
        if self.l515_started:
            self.l515_pipeline.stop()
        if self.d455_started:
            self.d455_pipeline.stop()
        cv2.destroyAllWindows()
        print("\nCamera cleanup complete")

if __name__ == "__main__":
    # while True:
    DualRealsenseCamera().start_cameras()