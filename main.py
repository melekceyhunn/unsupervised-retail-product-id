import cv2
import numpy as np
import torch
import cupy as cp # GPU Matris İşlemleri
from numba import njit #İşlemci Hızlandırma
import os
from ultralytics import YOLO
import sys
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel   # DINOv2 için

# --- BÖLÜM 1: DINOv2 BACKEND (FP16 OPTİMİZASYONLU) ---
class OptimizedMatcher:
    def __init__(self, model_name="facebook/dinov2-small"): 
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        print(f"Matcher başlıyor. Cihaz:{self.device}")
        if self.device=="cpu":
            print("cpu kullanılıyor, bu yavaş olabilir.")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            if self.device=="cuda":
                self.model.half()  # FP16 optimizasyonu sağlandı
                print("FP16 optimizasyonu sağlantı.")
        except Exception as e:
            print("Model yükleme hatası:", e)
            sys.exit()
        self.referace_embeddings_gpu = None
        self.referance_names = []
        self.is_ready = False
    def load_reference_folder(self, folder_path):
       # urun_kutuphanesi içindeki resimleri okur ve GPU matrisine dönüştürür.
        valid_exts = [".png", ".jpg", ".jpeg"] # desteklenen resim formatları jpg, jpeg, png, bmp
        temp_emb_list = []
        self.referance_names = []  
        if not os.path.exists(folder_path):
            print("Klasör bulunamadı:", folder_path)
            return
        print(f"Kütüphane taranıyor: {folder_path}...")
        files=sorted(os.listdir(folder_path))
        for f in files:
            name,ext=os.path.splitext(f)
            if ext.lower() not in valid_exts:
                continue
            path=os.path.join(folder_path,f)
            image=cv2.imread(path)
            if image is None:
                continue
            #BGR to RGB (DINOv2 modeli RGB formatında eğitilmiştir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs=self.processor(images=image,return_tensors="pt").to(self.device)
            with torch.no_grad():
                if self.device=="cuda":
                    inputs = {k: v.half() for k, v in inputs.items()}
                    outputs=self.model(**inputs) # Özellik çıkarımı
                else:
                    outputs=self.model(**inputs)
            feat=outputs.last_hidden_state[:,0,:]
            feat=feat / feat.norm(dim=-1,keepdim=True) # L2 Normalizasyonu
            temp_emb_list.append(feat.cpu().numpy()) # Cupy için float32 yapıldı
            self.referance_names.append(name)
            print(f"yüklendi:{f}")
        if temp_emb_list:
            all_embeddings=np.vstack(temp_emb_list)
            if self.device=="cuda":
                self.referace_embeddings_gpu=cp.array(all_embeddings)
            else:
                self.referace_embeddings_gpu=torch.tensor(all_embeddings)
            self.is_ready=True  
            print("Tüm referanslar görüntüleri yüklendi. Toplam:", len(self.referance_names))
        else:
            print("Klasör boş veya resim bulunamadı.")
    def find_best_matches(self, crop_images):
        # Gelen resimleri eşleştirir ve en iyi sonuçları döner.
        if not crop_images or not self.is_ready:
            return [], []
        inputs=self.processor(images=crop_images,return_tensors="pt").to(self.device)
        with torch.no_grad():
            if self.device=="cuda":
                inputs={k:v.half() for k,v in inputs.items()}
                outputs=self.model(**inputs)
            else:
                outputs=self.model(**inputs)
        batch_feat=outputs.last_hidden_state[:,0,:]
        batch_feat=batch_feat / batch_feat.norm(dim=-1,keepdim=True)   
        if self.device=="cuda":
            batch_gpu=cp.array(batch_feat.cpu().float().numpy(), dtype=cp.float32)
            similarity_matrix=cp.dot(batch_gpu, self.referace_embeddings_gpu.T)
            best_indices=cp.argmax(similarity_matrix, axis=1)
            best_scores=cp.max(similarity_matrix, axis=1)
            labels=[self.referance_names[i] for i in cp.asnumpy(best_indices)]
            return labels, cp.asnumpy(best_scores)
        else:
            batch_cpu=batch_feat.cpu().float().numpy()
            similarity_matrix=np.dot(batch_cpu, self.referace_embeddings_gpu.T)
            best_indices=np.argmax(similarity_matrix, axis=1)
            best_scores=np.max(similarity_matrix, axis=1)
            labels=[self.referance_names[i] for i in best_indices]
            return labels, best_scores
        
# --- BÖLÜM 2: HAFIZA (TRACKER) ---
class ProductMemory:
    def __init__(self, persistence=5,iou_threshold=0.5):
        self.memory=[]
        self.persistence=persistence
        self.iou_threshold=iou_threshold
    def update(self, current_detections):
        for item in self.memory:
            item['life']-=1
        if not current_detections:
            self.memory=[item for item in self.memory if item['life']>0]
            return self.memory
        new_memory=[]
        matched_indices=set()
        for detect_box, detect_label, det_score in current_detections:
            matched=False
            for i, memory_item in enumerate(self.memory):
                if i in matched_indices:
                    continue
                iou=self.compute_iuo(detect_box, memory_item['box'])
                if iou>=self.iou_threshold:
                    new_memory.append({
                        'box': detect_box,
                        'label': detect_label,
                        'life': self.persistence
                    })
                    matched_indices.add(i)
                    matched = True
                    break
            if not matched:
                new_memory.append({
                    'box': detect_box,
                    'label': detect_label,
                    'life': self.persistence
                })
        for i, memory_item in enumerate(self.memory):
            if i not in matched_indices and memory_item['life']>0:
                new_memory.append(memory_item)
        self.memory=new_memory
        return self.memory
    # Kesişim oranı hesaplaması
    @staticmethod
    @njit
    def compute_iuo(boxa, boxb):
        xa=max(boxa[0], boxb[0])
        ya=max(boxa[1], boxb[1])
        xb=min(boxa[2], boxb[2])
        yb=min(boxa[3], boxb[3])
        inter_area=max(0, xb - xa) * max(0, yb - ya)
        boxa_area=(boxa[2]-boxa[0]) * (boxa[3]-boxa[1])
        boxb_area=(boxb[2]-boxb[0]) * (boxb[3]-boxb[1])
        return inter_area / float(boxa_area + boxb_area - inter_area + 1e-6)

# --- BÖLÜM 3: ANA İŞLEM ---
def main():
    video_path="planogram.mp4"  # GİRİŞ VİDEOSU
    reference_folder="urun_kutuphanesi"  # REFERANS GÖRÜNTÜ KLASÖRÜ
    yolo_model_path="best.pt"  # YOLOv11  MODELİ
    detection_threshold=0.65  # TESPİT EŞİĞİ
    frame_skip=1  # HER frame_skip KAREDE BİR İŞLEM YAPILIR
    show_window = False if os.environ.get("NO_DISPLAY") else True 
    
    # hangi ortamda çalışacak (local/docker)
    is_running_in_docker = os.environ.get("DOCKER_ENV") == "true"
    
    # DINOv2 Matcher başlatma
    matcher=OptimizedMatcher(model_name="facebook/dinov2-small")
    matcher.load_reference_folder(reference_folder)
    tracker=ProductMemory(persistence=frame_skip+5)
    if not matcher.is_ready:
        print("Matcher hazır değil, çıkılıyor.")
        return
    print(f"YOLO modeli yükleniyor:{yolo_model_path}")
    detector=YOLO(yolo_model_path)
    if not os.path.exists(video_path):
        print("Video dosyası bulunamadı:", {video_path})
        return
    
    cap=cv2.VideoCapture(video_path)
    raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if is_running_in_docker:
        print("Docker ortamı!! Tespit işlemi için video 90 derece döndürülecek.")
        # Docker'da döndüreceğimiz için genişlik ve yükseklik yer değiştirir
        new_width = raw_height
        new_height = raw_width
        output_Video_path = "output_video_docker.avi"
        out=cv2.VideoWriter(
        output_Video_path,
        cv2.VideoWriter_fourcc(*'MJPG'),
        fps,
        (new_width, new_height)
    ) 
    else:
        print("Local ortam algılandı: Orijinal boyutlar korunuyor.")
        # Localde her şey normal
        new_width = raw_width
        new_height = raw_height
        output_Video_path = "output_video_local.mp4"
        out=cv2.VideoWriter(
        output_Video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (new_width, new_height)
    )     
    print("işlem başladı çıkmak için 'q' ya basın.")
    
    last_drawan_objects=[]
    frame_idx=0
    
    while True:
        ret,frame =cap.read()
        if not ret:
            break
        if is_running_in_docker:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        current_h, current_w = frame.shape[:2]

        frame_idx+=1
        if frame_idx % frame_skip ==0:
            use_halh=True if matcher.device=="cuda" else False
            # daha iyi görünmesi için conf değeri 0.15 yapıldı, küçük nesneler için
            results=detector(frame,conf=0.15,verbose=False,half=use_halh)
            
            if len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
            else:
                boxes = []
            
            cropes=[]
            raw_detections=[]
            
            for box in boxes:
                x1,y1,x2,y2=map(int,box)
                x1,y1=max(0,x1),max(0,y1)
                x2,y2=min(current_w,x2),min(current_h,y2)
                
                if (x2-x1) * (y2-y1) <10:
                    continue
                crop=frame[y1:y2,x1:x2]
                crop=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cropes.append(crop)
                raw_detections.append((x1,y1,x2,y2))
                
            current_valid_detections=[]
            if cropes:
                labels,scores=matcher.find_best_matches(cropes)
                for i,(label,score) in enumerate(zip(labels,scores)):
                    if score>=detection_threshold:
                        current_valid_detections.append((raw_detections[i], label, score))
            last_drawan_objects=tracker.update(current_valid_detections)
        
        for item in last_drawan_objects:
            if item['life']<=0:
                continue
            x1,y1,x2,y2=item['box']
            label=item['label']
            color=(0,255,0) # Yeşil bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,thickness=1)
            
            text=f"{label}"
            font_scale=0.4
            font_thick=1
            font=cv2.FONT_HERSHEY_SIMPLEX
            (tw,th),_=cv2.getTextSize(text,font,font_scale,font_thick)
            cv2.rectangle(frame,(x1,y1-th-4),(x1+tw,y1),color,-1)
            cv2.putText(frame,text,(x1,y1-4),font,font_scale,(0,0,0),font_thick)
            
        out.write(frame)
        
# --- BÖLÜM 4: CANLI GÖSTERİM ---   
        if show_window:         
            display_frame=cv2.resize(frame,(0, 0), fx=0.4, fy=0.4)
            cv2.imshow("Ürün Tanıma",display_frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
    cap.release()
    out.release()   
    if show_window:
        cv2.destroyAllWindows()
    print("işlem tamamlandı. Video kaydedildi:", {output_Video_path})
    
if __name__=="__main__":
    main()