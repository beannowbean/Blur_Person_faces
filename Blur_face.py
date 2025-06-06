# ===================================================================
# 1. 라이브러리 불러오기 (import)
# ===================================================================
import os 
import cv2 as cv
import numpy as np

# 우리가 설치한 Detectron2 라이브러리에서 필요한 기능들을 불러옵니다.
try:
    from detectron2.config import get_cfg 
    from detectron2.engine import DefaultPredictor 
    from detectron2.model_zoo import model_zoo 
except ImportError:
    print("오류: Detectron2 모듈을 찾을 수 없습니다. 'CatDetected' 가상 환경 설정을 확인해주세요.")
    exit()

# ===================================================================
# 2. 함수 정의
# ===================================================================

def setup_person_detector():
    """ Detectron2를 사용해서 '사람'을 찾아내는 예측 모델을 준비하고 생성하는 함수 """
    cfg = get_cfg()
    config_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg.MODEL.DEVICE = "cpu"
    predictor = DefaultPredictor(cfg)
    print("사람 탐지 모델(Detectron2) 준비 완료. (CPU 모드)")
    return predictor

# --- ⭐⭐⭐ 함수의 파라미터가 face_detector_net (DNN 모델)으로 변경되었습니다 ⭐⭐⭐ ---
def mosaic_faces_in_video(video_path, person_detector, face_detector_net, person_thresh, face_thresh):
    """ 동영상을 읽어서 사람은 Detectron2로, 얼굴은 DNN으로 찾아 모자이크 처리하는 메인 함수 """
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("동영상이 끝났습니다.")
            break
        
        result_frame = frame.copy()
        
        outputs = person_detector(frame)
        instances = outputs["instances"].to("cpu")
        person_boxes = instances.pred_boxes.tensor.numpy()
        person_classes = instances.pred_classes.numpy()
        person_scores = instances.scores.numpy()
        
        for i in range(len(instances)):
            score = person_scores[i]
            class_id = person_classes[i]
            
            if score > person_thresh and class_id == 0:
                box = person_boxes[i].astype(int)
                (startX, startY, endX, endY) = (box[0], box[1], box[2], box[3])
                
                person_roi = result_frame[startY:endY, startX:endX]
                if person_roi.shape[0] < 1 or person_roi.shape[1] < 1:
                    continue

                # --- ⭐⭐⭐ 2단계: Haar Cascade 대신 DNN 모델로 얼굴 검출! ⭐⭐⭐ ---
                (h_roi, w_roi) = person_roi.shape[:2]
                # ROI 이미지를 딥러닝 모델에 입력 가능한 형태로 변환(blob)
                blob = cv.dnn.blobFromImage(person_roi, 1.0, (300, 300), (104.0, 177.0, 123.0))
                # 얼굴 검출 신경망에 blob을 입력으로 설정
                face_detector_net.setInput(blob)
                # 얼굴 검출 실행
                detections = face_detector_net.forward()

                # ROI 안에서 찾은 얼굴들에 대해 반복
                for j in range(0, detections.shape[2]):
                    face_confidence = detections[0, 0, j, 2]

                    # 얼굴일 확률이 우리가 정한 기준보다 높으면
                    if face_confidence > face_thresh:
                        # 얼굴의 바운딩 박스 좌표를 ROI 기준으로 계산
                        face_box = detections[0, 0, j, 3:7] * np.array([w_roi, h_roi, w_roi, h_roi])
                        (fx, fy, f_endX, f_endY) = face_box.astype("int")
                        
                        # 박스 좌표가 ROI를 벗어나지 않도록 보정
                        fx, fy = max(0, fx), max(0, fy)
                        f_endX, f_endY = min(w_roi, f_endX), min(h_roi, f_endY)
                        
                        # 3단계: 찾아낸 얼굴 영역에만 모자이크 적용
                        face_sub_roi = person_roi[fy:f_endY, fx:f_endX]
                        if face_sub_roi.shape[0] < 1 or face_sub_roi.shape[1] < 1: continue
                        
                        pixel_level = 15
                        small_face = cv.resize(face_sub_roi, (max(1, int((f_endX-fx)/pixel_level)), max(1, int((f_endY-fy)/pixel_level))), interpolation=cv.INTER_LINEAR)
                        mosaic_face = cv.resize(small_face, ((f_endX-fx), (f_endY-fy)), interpolation=cv.INTER_NEAREST)
                        
                        person_roi[fy:f_endY, fx:f_endX] = mosaic_face

        cv.imshow("Face Mosaic Project (DNN Ver.)", result_frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv.destroyAllWindows()

# ===================================================================
# 3. 메인 코드 실행 부분
# ===================================================================
if __name__ == '__main__':
    # --- 사용자 설정 영역 ---
    PERSON_MODEL_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    PERSON_CONF_THRESHOLD = 0.6
    
    # ⭐ 얼굴 검출 모델을 Caffe DNN 모델로 설정
    FACE_PROTO = "/Users/ihyeonbin/Documents/GitHub/Blur_Person_faces/models/deploy.prototxt.txt"
    FACE_MODEL = "/Users/ihyeonbin/Documents/GitHub/Blur_Person_faces/models/res10_300x300_ssd_iter_140000.caffemodel"
    FACE_CONF_THRESHOLD = 0.5

    VIDEO_PATH = "/Users/ihyeonbin/Documents/GitHub/Blur_Person_faces/sample_video.mp4"
    # ---------------------------

    # 필요한 파일들이 모두 있는지 확인
    if not all(os.path.exists(p) for p in [FACE_PROTO, FACE_MODEL, VIDEO_PATH]):
        print("오류: 모델 파일 또는 비디오 파일 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        try:
            # 1. 사람을 찾아낼 Detectron2 모델을 준비
            person_detector = setup_person_detector()
            
            # --- ⭐⭐⭐ Haar Cascade 대신 DNN 모델을 로드합니다! ⭐⭐⭐ ---
            face_detector_net = cv.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
            
            print("모든 모델 준비 완료! 업그레이드된 얼굴 모자이크 프로젝트를 시작합니다...")
            
            # 3. 메인 함수를 실행합니다!
            mosaic_faces_in_video(
                video_path=VIDEO_PATH,
                person_detector=person_detector,
                face_detector_net=face_detector_net, # DNN 모델을 전달
                person_thresh=PERSON_CONF_THRESHOLD,
                face_thresh=FACE_CONF_THRESHOLD
            )
        except Exception as e:
            print("오류가 발생했습니다:", e)