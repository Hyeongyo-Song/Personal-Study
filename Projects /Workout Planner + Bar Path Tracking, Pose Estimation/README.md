<p align="center">
  <img src="https://github.com/user-attachments/assets/2697083c-914c-460e-baa5-8746a7174d47"  width="200" height="400"/>
  <img src="https://github.com/user-attachments/assets/39278b9d-a8dc-4345-8582-4b979b72d6fd"  width="200" height="400"/>
  <img src="https://github.com/user-attachments/assets/2f5aecc3-53f1-43b7-928f-224df977e2b8"  width="200" height="400"/>
</p>



운동을 계획하고, 계획된 운동 수행을 돕는 Workout Planner Application입니다.
또한 Tencent의 YOLOv7-tiny, Google의 ML Kit를 활용하여 바벨의 궤적과 사용자의 자세를 Skeleton으로 시각화하는 기능이 있습니다.

기존에도 AI를 활용한 안드로이드용 Barbell Tracking이나 Pose Estimation이 존재했지만
실시간 카메라에서의 Detection만 지원하거나, Barbell Tracking과 Pose Estimation 둘 중 오직 하나만 지원하는 문제가 있었습니다.

본 프로젝트에서는 기존의 어플리케이션들과 차별성을 두기 위해

1. 안드로이드 환경에서 작동 가능해야 한다.
2. 실시간 감지 뿐 아니라, 사용자가 갤러리에서 선택한 동영상을 분석하는 것 또한 가능해야 한다.
3. Object Detection과 Pose Estimation을 모두 지원해야 한다.
의 3가지에 초점을 두었습니다.

결과적으로,

1. MediaExtractor API가 영상의 프레임을 분해한다.
2. 분해된 프레임의 타입을 Bitmap으로 Convert한다. (YUV 컬러로 Convert 되므로 RGB 컬러로 변환하는 작업도 함께 수행해야 한다.)
3. 각 프레임에 Object Detection, Pose Estimation을 수행한 뒤 그 결과를 그린다.
4. 모든 프레임 처리가 완료되었다면 MediaMuxer API를 사용하여 하나의 영상으로 병합하여 반환한다.
라는 로직을 구현함으로써 프로젝트를 완수했습니다.

물론, 현재는 뼈대만 만들어진 상황입니다. 차차 발전시켜 나갈 예정입니다.

더 자세한 내용은 PPT를 참고해주세요 !
