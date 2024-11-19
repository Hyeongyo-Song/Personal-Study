# 🏋️ Workout Planner Application

> 운동을 계획하고 수행을 돕는 Workout Planner Application입니다.  
> Tencent의 YOLOv7-tiny와 Google의 ML Kit를 활용하여 바벨의 궤적과 사용자의 자세를 Skeleton으로 시각화하는 기능을 제공합니다.

## 📸 주요 스크린샷
<p align="center">
  <img src="https://github.com/user-attachments/assets/2697083c-914c-460e-baa5-8746a7174d47" width="200" height="400" alt="Screenshot 1"/>
  <img src="https://github.com/user-attachments/assets/39278b9d-a8dc-4345-8582-4b979b72d6fd" width="200" height="400" alt="Screenshot 2"/>
  <img src="https://github.com/user-attachments/assets/2f5aecc3-53f1-43b7-928f-224df977e2b8" width="200" height="400" alt="Screenshot 3"/>
</p>



## 📝 프로젝트 개요

### **기존 애플리케이션의 문제점**
- **제한적인 기능**: 실시간 감지만 지원하거나, Barbell Tracking과 Pose Estimation 중 하나만 지원.  

### **본 프로젝트의 차별점**
1. **안드로이드 환경에서 작동**: 모든 기능이 모바일 환경에서 실행 가능.  
2. **동영상 분석 지원**: 실시간 감지뿐만 아니라, 갤러리에서 선택한 동영상도 분석 가능.  
3. **다기능 지원**: Object Detection과 Pose Estimation을 모두 수행.  



## ⚙️ 구현 로직

1. **프레임 분해**: `MediaExtractor API`로 동영상을 프레임 단위로 분해.  
2. **프레임 변환**: 분해된 프레임을 Bitmap으로 변환하며, YUV → RGB 컬러 변환 작업 수행.  
3. **Object Detection & Pose Estimation**: 각 프레임에 YOLOv7-tiny와 ML Kit를 사용해 감지 및 Skeleton 시각화 수행.  
4. **영상 병합**: 모든 처리가 완료된 프레임을 `MediaMuxer API`를 사용해 하나의 영상으로 병합하여 반환.  



## 🚀 프로젝트 진행 상황

현재는 뼈대만 완성된 상태이며, 지속적으로 기능을 개선하고 확장해 나갈 예정입니다.  
**자세한 내용은 프로젝트 PDF**를 참고하세요.  



## 📂 프로젝트 파일

- **전체 용량**: 약 8GB (APK 포팅 시 약 300MB).  
- 본 레포지토리에 업로드하지 않았습니다.  
- 필요 시 아래 이메일로 문의해주세요.



## 📧 문의
- **Email:** songhg0321@naver.com || plx17842@gmail.com
