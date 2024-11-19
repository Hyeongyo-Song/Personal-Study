import cv2
import torch
import numpy as np
import librosa
import librosa.feature
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from moviepy.editor import VideoFileClip
from scipy.io.wavfile import write
import os
from timm import create_model
import warnings

class VideoToAudioModel(nn.Module):
    def __init__(self, output_size=(128, 128)): # 출력 크기 (128, 128)
        super(VideoToAudioModel, self).__init__()
        self.output_size = output_size
        self.encoder = create_model('convnext_tiny', pretrained=True, num_classes=256) # Convnext 모델 사용. 최신 트렌드를 반영하기 위함 ...
        self.rnn = nn.GRU(256, 256, batch_first=True, num_layers=2)  # Gated Recurrent Unit 사용. 경량화를 위해.
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_size[0] * self.output_size[1]),
        )

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        x = x.reshape(batch_size * depth, channels, height, width)  # [batch_size * depth, C, H, W]
        x = self.encoder(x)
        x = x.view(batch_size, depth, -1)  # [batch_size, depth, 256]
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.decoder(x)
        return x.view(batch_size, *self.output_size)  # 출력 크기 (Batch Size, 128, 128)

class VideoDataset(Dataset):
    def __init__(self, video_paths, audio_paths, target_size=(128, 128), max_frames=16):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.target_size = target_size
        self.max_frames = max_frames

    def preprocess_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        frame_count = 0
        while cap.isOpened() and frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, self.target_size)
            frame_tensor = torch.tensor(frame).permute(2, 0, 1).float() / 255.0  # Regularization
            if frame_tensor.size(0) == 1:
                frame_tensor = frame_tensor.repeat(3, 1, 1)
            frames.append(frame_tensor)
            frame_count += 1
        cap.release()
        frames_tensor = torch.stack(frames)  # [D, C, H, W]
        frames_tensor = frames_tensor.permute(1, 0, 2, 3)  # [C, D, H, W]
        return frames_tensor, fps

    def preprocess_audio(self, audio_path, fps):
        y, sr = librosa.load(audio_path, sr=None)
        hop_length = int(sr / fps)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        log_spectrogram = torch.tensor(log_spectrogram).unsqueeze(0)
      
        log_spectrogram = torch.nn.functional.interpolate(
            log_spectrogram.unsqueeze(0), size=(128, 128), mode='bilinear'
        ).squeeze(0)  # [1, 128, 128]
        return log_spectrogram

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        audio_path = self.audio_paths[idx]
        frames, fps = self.preprocess_video(video_path)
        spectrogram = self.preprocess_audio(audio_path, fps)
        return frames, spectrogram


if __name__ == "__main__":
    video_path = "C:/Users/samsung/PycharmProjects/GraduateSchoolProject/Project/video/"
    audio_output_path = "C:/Users/samsung/PycharmProjects/GraduateSchoolProject/Project/audio/"
    os.makedirs(audio_output_path, exist_ok=True)

    video_files = [os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.mp4')]
    audio_files = [os.path.join(audio_output_path, f"output{idx}.wav") for idx, _ in enumerate(video_files)]

    for idx, video_file in enumerate(video_files):
        video_clip = VideoFileClip(video_file)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_files[idx])
        video_clip.close()
        audio_clip.close()

    dataset = VideoDataset(video_files, audio_files, max_frames=16)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VideoToAudioModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  
    for epoch in range(1000):
        model.train()
        for frames, spectrograms in dataloader:
            frames = frames.to(device)
            spectrograms = spectrograms.to(device)

            if frames.size(2) != spectrograms.size(1):
                spectrograms = spectrograms.expand(frames.size(0), frames.size(2), -1, -1)

            outputs = model(frames)
            loss = criterion(outputs, spectrograms)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}")

    input_video_path = "C:/Users/samsung/PycharmProjects/GraduateSchoolProject/Project/input/"
    output_audio_path = "C:/Users/samsung/PycharmProjects/GraduateSchoolProject/Project/output/"
    os.makedirs(output_audio_path, exist_ok=True)

    input_video_files = [os.path.join(input_video_path, f) for f in os.listdir(input_video_path) if f.endswith('.mp4')]

    model.eval()
    for idx, video_file in enumerate(input_video_files):
        frames, fps = dataset.preprocess_video(video_file)
        frames = frames.unsqueeze(0).to(device)  # [1, C, D, H, W]

        with torch.no_grad():
            predicted_spectrogram = model(frames)

        predicted_spectrogram = torch.clamp(predicted_spectrogram, min=-80.0, max=0.0)

        hop_length = int(22050 / fps)
        predicted_spectrogram = predicted_spectrogram.squeeze(0).cpu().numpy()
        spectrogram_power = librosa.db_to_power(predicted_spectrogram)
        audio = librosa.feature.inverse.mel_to_audio(spectrogram_power, sr=22050, hop_length=hop_length)

      # 오디오 길이를 원본 비디오와 동기화
        video_clip = VideoFileClip(video_file)
        target_duration = video_clip.duration
        audio_duration = librosa.get_duration(y=audio, sr=22050)
        if audio_duration < target_duration:
            audio = librosa.util.fix_length(audio, size=int(target_duration * 22050), mode='constant')

        output_audio_file = os.path.join(output_audio_path, f"generated_audio_{idx}.wav")
        write(output_audio_file, 22050, (audio * 32767).astype(np.int16))
        print(f"Generated audio for {video_file} saved to {output_audio_file}")
