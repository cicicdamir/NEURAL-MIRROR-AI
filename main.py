import cv2
import insightface
from insightface.app import FaceAnalysis
import os
import sys
import numpy as np
from gfpgan import GFPGANer
import subprocess

# --- KONFIGURACIJA ---
FACE_INPUT = "face.jpg"
VIDEO_INPUT = "video.mp4"
MODEL_SWAPPER = "inswapper_128.onnx"
GFPGAN_MODEL_PATH = "GFPGANv1.4.pth"
# ---------------------

def run_pro_swap():
    # 1. GPU Inicijalizacija za AI (ovo radi jer koristi CUDA, ne NVENC)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    swapper = insightface.model_zoo.get_model(MODEL_SWAPPER, providers=['CUDAExecutionProvider'])
    restorer = GFPGANer(model_path=GFPGAN_MODEL_PATH, upscale=1, arch='clean', channel_multiplier=2, device='cuda')

    # 2. Priprema lica
    face_img = cv2.imread(FACE_INPUT)
    source_faces = app.get(face_img)
    if not source_faces:
        print("Greska: Lice nije pronadjeno.")
        return
    source_face = sorted(source_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)[0]

    cap = cv2.VideoCapture(VIDEO_INPUT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. FFmpeg Pipe - Koristimo 'libx264' ali na najlaksi moguci nacin
    output_temp = "output_raw.mp4"
    command = [
        'ffmpeg', '-y', 
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', f'{fps}',
        '-i', '-', 
        '-c:v', 'libx264', 
        '-preset', 'ultrafast', # Najvaznije za brzinu na procesoru
        '-crf', '20',           # Dobar kvalitet, manji fajl
        '-pix_fmt', 'yuv420p',  # OBAVEZNO: Ovo resava High 4:4:4 problem i ubrzava 3x
        output_temp
    ]
    
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    print(">>> Obrada pocela (CPU Enkoding + GPU AI)...")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        target_faces = app.get(frame)
        
        # Radimo samo najveće lice u kadru
        if target_faces:
            face = max(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
            frame = swapper.get(frame, face, source_face, paste_back=True)
            # Restauracija samo ako je neophodno (ovde ukljuceno za kvalitet)
            _, _, frame = restorer.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True)

        process.stdin.write(frame.tobytes())
        
        frame_count += 1
        if frame_count % 20 == 0:
            sys.stdout.write(f"\rProgres: {(frame_count/total_frames)*100:.1f}% | FPS: {fps}")
            sys.stdout.flush()

    cap.release()
    process.stdin.close()
    process.wait()

    # 4. Spajanje audia
    os.system(f'ffmpeg -i {output_temp} -i {VIDEO_INPUT} -c copy -map 0:v:0 -map 1:a:0? -shortest FINAL_RESULT.mp4 -y')
    if os.path.exists(output_temp): os.remove(output_temp)
    print("\n>>> GOTOVO!")

if __name__ == "__main__":
    run_pro_swap()
