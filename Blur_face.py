# ===================================================================
# 1. 라이브러리 불러오기 (import)
# ===================================================================
# 파이썬의 기본적인 기능(경로 설정 등)을 사용하기 위한 라이브러리
import os 
# 이미지 처리 및 비디오를 다루기 위한 OpenCV 라이브러리를 cv 라는 별칭으로 불러옵니다.
import cv2 as cv
# 숫자 데이터, 특히 행렬(이미지)을 쉽게 다루기 위한 NumPy 라이브러리
import numpy as np

# 우리가 설치한 Detectron2 라이브러리에서 필요한 기능들을 불러옵니다.
try:
    # 모델의 기본 설정을 가져오는 기능
    from detectron2.config import get_cfg 
    # 설정에 따라 예측을 실행하는 "예측 전문가"를 만드는 기능
    from detectron2.engine import DefaultPredictor 
    # 미리 학습된 모델들의 목록을 관리하는 기능
    from detectron2.model_zoo import model_zoo 
except ImportError:
    # 만약 Detectron2가 제대로 설치되지 않았다면 에러 메시지를 보여주고 종료합니다.
    print("오류: Detectron2 모듈을 찾을 수 없습니다. 'CatDetected' 가상 환경 설정을 확인해주세요.")
    exit()

# ===================================================================
# 2. 함수 정의
# ===================================================================

def setup_person_detector():
    """
    Detectron2를 사용해서 '사람'을 찾아내는 예측 모델을 준비하고 생성하는 함수
    """
    # Detectron2의 기본 설정을 불러옵니다. (일종의 기본 레시피)
    cfg = get_cfg()
    
    # 우리가 사용할 모델의 레시피 파일로 설정을 덮어씌웁니다.
    # "RetinaNet"은 속도와 정확도의 균형이 좋은 모델입니다.
    config_file = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    # Model Zoo(모델 동물원)에서 미리 학습된 모델의 '지식' 파일(가중치)을
    # 인터넷으로 다운로드할 주소를 설정합니다. (첫 실행 시에만 다운로드)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    
    # 모델이 결과를 알려줄 때, "이건 80% 이상 확신하는 것만 알려줘" 라고 기준점을 정해줍니다.
    # 이 기준이 너무 높으면 객체를 놓칠 수 있고, 너무 낮으면 잘못된 객체를 찾을 수 있습니다.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    
    # 우리 맥북의 GPU(MPS)에서 에러가 발생했었으므로, 안정적인 CPU를 사용하도록 설정합니다.
    cfg.MODEL.DEVICE = "cpu"
    
    # 위에서 만든 모든 설정을 바탕으로, 이미지만 넣으면 바로 결과를 내어주는 '예측 전문가'를 만듭니다.
    predictor = DefaultPredictor(cfg)
    
    print("사람 탐지 모델(Detectron2) 준비 완료. (CPU 모드)")
    return predictor

def mosaic_faces_in_video(video_path, person_detector, face_detector, person_thresh, face_thresh):
    """
    동영상을 읽어서 사람을 찾고, 그 사람의 얼굴만 찾아 모자이크 처리하는 메인 함수
    """
    # 동영상 파일을 불러옵니다.
    cap = cv.VideoCapture(video_path)
    # 동영상을 제대로 불러왔는지 확인합니다.
    if not cap.isOpened():
        print(f"오류: 동영상 파일을 열 수 없습니다: {video_path}")
        return

    # 동영상 창이 열려있는 동안 계속 반복합니다.
    while True:
        # 동영상에서 한 프레임(한 장의 이미지)을 읽어옵니다.
        ret, frame = cap.read()
        # 더 이상 읽어올 프레임이 없으면(동영상이 끝나면) 반복을 멈춥니다.
        if not ret:
            print("동영상이 끝났습니다.")
            break
        
        # 원본 프레임을 복사해서, 이 복사본 위에 그림을 그릴 겁니다.
        result_frame = frame.copy()
        
        # --- 1단계: 사람 검출 ---
        # 준비된 '사람 예측 전문가'에게 현재 프레임을 보여주고 사람을 찾아달라고 요청합니다.
        outputs = person_detector(frame)
        
        # 예측 결과에서 필요한 정보(박스 좌표, 클래스, 점수)를 추출합니다.
        instances = outputs["instances"].to("cpu")
        person_boxes = instances.pred_boxes.tensor.numpy()
        person_classes = instances.pred_classes.numpy()
        person_scores = instances.scores.numpy()
        
        # 찾은 '사람' 객체들 각각에 대해 반복합니다.
        for i in range(len(instances)):
            # 이 객체가 '사람'일 확률(점수)을 가져옵니다.
            score = person_scores[i]
            # 이 객체가 어떤 종류(클래스)인지 가져옵니다. (COCO 데이터셋에서 '사람'은 0번)
            class_id = person_classes[i]
            
            # 만약 이 객체가 '사람'이고, 그 확률이 우리가 정한 기준점보다 높다면
            if score > person_thresh and class_id == 0:
                # 사람의 바운딩 박스 좌표를 가져옵니다.
                box = person_boxes[i].astype(int)
                (startX, startY, endX, endY) = (box[0], box[1], box[2], box[3])
                
                # 사람 영역만 잘라냅니다. (이걸 '관심 영역' 또는 ROI 라고 합니다)
                person_roi = result_frame[startY:endY, startX:endX]
                
                # ROI가 정상적인 크기인지 확인합니다.
                if person_roi.shape[0] < 1 or person_roi.shape[1] < 1:
                    continue

                # --- 2단계: 얼굴 검출 ---
                # 잘라낸 사람 영역을 흑백으로 바꿔서 '얼굴 검출 전문가'에게 보여줍니다.
                gray_roi = cv.cvtColor(person_roi, cv.COLOR_BGR2GRAY)
                # 얼굴 검출을 실행합니다.
                faces_in_roi = face_detector.detectMultiScale(gray_roi, 1.1, 5, minSize=(30, 30))

                # 찾은 '얼굴'들 각각에 대해 반복합니다.
                for (fx, fy, fw, fh) in faces_in_roi:
                    # 얼굴 영역만 더 작게 잘라냅니다.
                    face_sub_roi = person_roi[fy:fy+fh, fx:fx+fw]
                    if face_sub_roi.shape[0] < 1 or face_sub_roi.shape[1] < 1:
                        continue
                    
                    # --- 3단계: 모자이크 처리 ---
                    # 모자이크 강도를 정합니다. (숫자가 작을수록 모자이크가 커져요)
                    pixel_level = 20
                    # 얼굴 이미지를 아주 작게 축소합니다.
                    small_face = cv.resize(face_sub_roi, (max(1, int(fw/pixel_level)), max(1, int(fh/pixel_level))), interpolation=cv.INTER_LINEAR)
                    # 축소된 이미지를 다시 원래 크기로 확대합니다. (이때 픽셀이 뭉개지면서 모자이크 효과 발생)
                    mosaic_face = cv.resize(small_face, (fw, fh), interpolation=cv.INTER_NEAREST)
                    
                    # 모자이크 처리된 얼굴을 원래 사람 영역의 해당 위치에 다시 붙여넣습니다.
                    person_roi[fy:fy+fh, fx:fx+fw] = mosaic_face

        # 최종 결과물을 화면에 보여줍니다.
        cv.imshow("Face Mosaic Project", result_frame)

        # 키보드에서 'q' 키를 누르면 반복을 멈추고 프로그램을 종료합니다.
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # 사용했던 모든 자원을 해제합니다.
    cap.release()
    cv.destroyAllWindows()

# ===================================================================
# 3. 메인 코드 실행 부분
# ===================================================================
# 이 스크립트 파일을 직접 실행했을 때만 아래 코드가 동작합니다.
if __name__ == '__main__':
    # --- 사용자 설정 영역 ---
    # 1. 사람 검출 모델 (Detectron2)
    PERSON_MODEL_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    PERSON_CONF_THRESHOLD = 0.6
    
    # 2. 얼굴 검출 모델 (OpenCV DNN - Caffe)
    # 이 파일들은 미리 다운로드 받아두셔야 합니다!
    FACE_PROTO = "/Users/ihyeonbin/HumanIsCat/models/deploy.prototxt.txt"
    FACE_MODEL = "/Users/ihyeonbin/HumanIsCat/models/res10_300x300_ssd_iter_140000.caffemodel"
    FACE_CONF_THRESHOLD = 0.5

    # 3. 사용할 동영상 파일 경로
    VIDEO_PATH = "/Users/ihyeonbin/DNN_Project/Blur_Person_faces/sample_video.mp4"
    
    # 4. 사용할 OpenCV의 Haar Cascade 얼굴 검출기 모델 파일 경로
    HAAR_CASCADE_PATH = os.path.join(cv.data.haarcascades, 'haarcascade_frontalface_default.xml')
    # ---------------------------

    # 필요한 파일들이 모두 있는지 확인합니다.
    if not all(os.path.exists(p) for p in [FACE_PROTO, FACE_MODEL, VIDEO_PATH, HAAR_CASCADE_PATH]):
        print("오류: 모델 파일 또는 비디오 파일 경로를 찾을 수 없습니다. 경로를 확인해주세요.")
    else:
        try:
            # 1. 사람을 찾아낼 Detectron2 모델을 준비합니다.
            person_detector = setup_person_detector()
            
            # 2. 얼굴을 찾아낼 OpenCV Haar Cascade 모델을 준비합니다.
            face_detector = cv.CascadeClassifier(HAAR_CASCADE_PATH)
            
            print("모든 모델 준비 완료! 얼굴 모자이크 프로젝트를 시작합니다...")
            
            # 3. 메인 함수를 실행합니다!
            mosaic_faces_in_video(
                video_path=VIDEO_PATH,
                person_predictor=person_detector,
                face_detector=face_detector,
                person_thresh=PERSON_CONF_THRESHOLD,
                face_thresh=0 # Haar Cascade는 별도 임계값 없으므로 0으로 둠
            )

        except Exception as e:
            # 만약 실행 중 다른 에러가 발생하면, 어떤 에러인지 알려줍니다.
            print("오류가 발생했습니다:", e)