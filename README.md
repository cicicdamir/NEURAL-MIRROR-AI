# 🧠 NEURAL MIRROR | AI Face Swapper

A high-speed, CUDA-optimized pipeline for identity transfer using **InsightFace** and facial restoration via **GFPGAN**.



## ⚡ Architecture & Performance
- **Neural Engine:** Powered by `insightface` (buffalo_l) for high-precision face detection and swapping.
- **Visual Fidelity:** Integrated `GFPGANv1.4` to restore facial details and skin textures.
- **Hardware Acceleration:** Native **CUDA** support for near real-time processing.
- **Video Pipeline:** Uses **FFmpeg piping** to bypass disk bottlenecks, ensuring maximum I/O performance.

## 🚀 Quick Start
1. `pip install -r requirements.txt`
2. Place `lice.jpg` (source) and `video.mp4` (target) in this directory.
3. Run: `python main.py`

## ⚖️ Ethical Disclaimer
This project is for research purposes only. Please respect privacy and ethical standards when working with synthetic media.
