# Lay3rs Print Farm - Vision Submodule

## Requirements & installation
### Libraries
* Python 3.9+,
* Picamera2/libcamera, 
* NumPy v1.24-2.0,
* OpenCV including the aruco module, v4.7-5.0,
* pupil_apriltags 1.0.4-2.0.
``` 
source ~/vision-venv/bin/activate
pip install -r requirements.txt
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera
```

### Camera setup

In the ```/boot/firmware/config.txt```, under **[all]** section add/modify these lines:
```
camera_auto_detect=0
dtoverlay=ov9281,cam0
dtoverlay=ov9281,cam1
```