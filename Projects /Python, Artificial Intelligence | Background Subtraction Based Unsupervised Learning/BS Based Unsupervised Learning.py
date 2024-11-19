import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

def create_feature_extractor():
    base_model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
    ])
    return base_model

def create_model():
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # binary classification
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def is_similar_to_learned_object(new_object_features, learned_features, threshold=0.7):
    similarities = cosine_similarity([new_object_features], [learned_features])
    return similarities[0][0] > threshold

def should_learn_new_object(new_object, feature_extractor, learned_features, threshold=0.7):
    new_object_features = feature_extractor.predict(np.expand_dims(new_object, axis=0))[0]
    if is_similar_to_learned_object(new_object_features, learned_features, threshold):
        print("새 객체가 기존 학습된 객체와 유사하여 학습합니다.")
        return True
    else:
        print("새 객체가 기존 학습된 객체와 다르므로 학습하지 않습니다.")
        return False

def save_learned_features(features, file_path):
    np.save(file_path, features)
    print(f"학습된 특징이 '{file_path}'에 저장되었습니다.")

def load_learned_features(file_path):
    features = np.load(file_path)
    print(f"학습된 특징이 '{file_path}'에서 성공적으로 로드되었습니다.")
    return features

def preprocess_frame(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    equalized_frame = cv2.equalizeHist(gray)
    
    resized_frame = cv2.resize(equalized_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=-1)

def train_with_similar_objects(video_path, feature_extractor, learned_features_path, model_save_path, is_first_training=False):
    cap = cv2.VideoCapture(video_path)
    back_subtractor = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=400, detectShadows=False)

    if not cap.isOpened():
        print(f"동영상을 열 수 없습니다: {video_path}")
        return

    learned_features = None
    if not is_first_training:
        learned_features = load_learned_features(learned_features_path)
    
    model = create_model()
    X, y = [], []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("동영상의 끝에 도달했습니다.")
            break

        processed_frame = preprocess_frame(frame)

        if is_first_training or should_learn_new_object(processed_frame, feature_extractor, learned_features):
            X.append(processed_frame)
            y.append(1)

    cap.release()

    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        
        model.fit(X, y, epochs=10)
        save_model(model, model_save_path)

        if is_first_training:
            learned_features = feature_extractor.predict(np.array(X))
            save_learned_features(learned_features, learned_features_path)
    else:
        print("유사한 객체가 없어 학습할 데이터가 없습니다.")

def draw_bounding_box_moving_objects(frame, fg_mask):
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # 노이즈 제거를 위해 최소 영역 크기 설정
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def detect_and_save_to_video(video_path, model, output_video_path, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    back_subtractor = cv2.createBackgroundSubtractorKNN(history=1000, dist2Threshold=400, detectShadows=False)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("객체 탐지 및 동영상 저장을 시작합니다.")
    frame_count = 0

    frame_batch = []
    processed_frames_batch = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("동영상의 끝에 도달했습니다.")
            break

        fg_mask = back_subtractor.apply(frame)
        processed_frame = preprocess_frame(frame)

        frame_batch.append(frame)
        processed_frames_batch.append(processed_frame)

        if len(processed_frames_batch) == batch_size:
            predictions = model.predict(np.array(processed_frames_batch))

            for i, prediction in enumerate(predictions):
                if prediction > 0.5:
                    frame_batch[i] = draw_bounding_box_moving_objects(frame_batch[i], fg_mask)

                out.write(frame_batch[i])

            frame_batch = []
            processed_frames_batch = []

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"{frame_count}개의 프레임을 처리했습니다.")

    if frame_batch:
        predictions = model.predict(np.array(processed_frames_batch))
        for i, prediction in enumerate(predictions):
            if prediction > 0.5:
                frame_batch[i] = draw_bounding_box_moving_objects(frame_batch[i], fg_mask)

            out.write(frame_batch[i])

    cap.release()
    out.release()
    print(f"결과 동영상이 '{output_video_path}'에 저장되었습니다.")

def save_model(model, file_path):
    model.save(file_path)
    print(f"모델이 '{file_path}'에 저장되었습니다.")

def load_model(file_path):
    model = tf.keras.models.load_model(file_path)
    print(f"모델이 '{file_path}'에서 성공적으로 로드되었습니다.")
    return model

if __name__ == "__main__":
    video_path = 'C:/datasets/MP4_6.mp4'
    model_save_path = 'C:/datasets/saved_model.h5'
    learned_features_path = 'C:/datasets/learned_features.npy'
    output_video_path = 'C:/datasets/output.mp4'

    feature_extractor = create_feature_extractor()

    mode = input("모드를 선택하세요 ('train' 또는 'detect'): ").strip()

    if mode == 'train':
        train_with_similar_objects(video_path, feature_extractor, learned_features_path, model_save_path, is_first_training=True)

    elif mode == 'detect':
        loaded_model = load_model(model_save_path)
        detect_and_save_to_video(video_path, loaded_model, output_video_path)

    else:
        print("올바른 모드를 입력하세요 ('train' 또는 'detect').")
