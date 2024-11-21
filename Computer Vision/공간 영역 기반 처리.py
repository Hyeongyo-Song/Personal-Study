# 화소 기반 처리 : 화소값 각각에 대해 여러 가지 연산 수행
# 공간 영역 기반 처리 : 영상처리의 대상이 하나의 화소가 아니라 그 화소와 주변 화소들의 밝기 값을 고려하여 대상 화소의 밝기값을 변환하는 처리.
    # 대상 화소 및 주변 화소의 구조 : 대상 픽셀을 중심으로 사각형으로 구성되며, 상하좌우 대칭을 위하여 홀수 크기를 갖는다. 필요한 경우 십자가 모양 등으로도 구성 가능.

# Convolution : 커널의 각 요소와 대응하는 입력 픽셀 값을 곱해서 모두 합하는 것. 공간 기반 영상 처리의 대표적인 연산, 영상 처리의 많은 필터링 기술들이 컨볼루션에 기반, 딥러닝 CNN에도 컨볼루션이 사용됨.
    # Convolution 연산 : 마스크는 순차적으로 우측으로 이동하고, 해당 라인의 끝에 도달하면 다음라인으로 이동 동작을 하면서 화소마다 컨볼루션 연산을 반복.

# 경계선 처리 : 경계선에 위치하는 픽셀인 경우 마스크가 놓이는 곳이 입력 영상의 경계를 벗어나는 문제를 해결
    # 1. 경계선 무시 : 마스크가 항상 입력 영상의 내부에 존재하도록 처리. 출력 영상의 크기가 입력 영상의 크기보다 줄어들거나 경계선 부분은 영상처리가 되지 않음.
    # 2. Zero Padding : 마스크가 위치한 곳에 입력 영상의 밝기 값이 없는 경우 0으로 처리. 출력 영상의 크기가 입력영상의 크기가 동일하게 유지됨. 관심의 대상은 영상의 중심에 위치함.(경계선 부분은 주요 관심 대상물이 없다.)
    # 3. Mirroring : 마스크가 위치한 곳에 입력 영상의 밝기 값이 없는 경우 인접한 밝기값으로 채우는 방식. 출력 영상의 크기가 입력 영상의 크기와 동일하게 유지. 관심의 대상은 영상의 중심에 위치함.(경계선 부분은 주요 관심 대상물이 없다.)

# 함수 : cv2.filter2D(mat, ddepth, kernel[, dst, anchor, delta, borderType])
    # ddepth : 출력 영상의 dtype(데이터 타입), -1이면 입력 영상과 동일함. CV_8U, 16U, 16S, 32F, 64F
    # kerner : 컨볼루션 커널 matrix
    # borderType : 경계선 보정 방법 지정 / https://wikidocs.net/231809 / cv2.BORDER_CONSTANT, REFLECT, REFLECT_101, REPLICATE, BORDER_WRAP, BORDER_ISOLATED

# 블러링(Blurring) : 영상에서 화소값이 급격하게 변하는 부분을 감소시켜 점진적으로 변하게 함으로써 영상이 전체적으로 부드러운 느낌이 나게 하는 기술.
    # 마스크를 이용하여 영상을 흐릿하게 처리함. 노이즈 제거에 탁월, 사진의 아웃 포커싱, 포토샵의 뽀샤시 효과를 연출.
    # 종류 : 평균, 메디안, 가우시안 블러

# 평균값 필터링(Averaging Blur)
    # 각 화소의 값을 주변 픽셀들과의 평균값으로 결정.
    # 3*3 필터, 5*5 필터가 평균 필터 사이즈, 필터가 클수록 흐릿해짐.
    # 필터의 합이 1이 되어야 출력 영상의 화소값(밝기) 변화가 없음.
    # cv2.filter2D(), boxFilter(), blur() 등 블러 처리를 위한 함수들 존재.
    # 공통부분을 합한후 필터의 사이즈로 나눠줌으로 동작함.

# 미디언 블러링(Median Blurring)
    # 마스크 범위 원소 중 중간 값을 출력 화소로 결정하는 방식
    # 새로운 값이 아닌 기존 픽셀의 값을 사용
    # Salt & Pepper noise(점잡음) 제거에 효과적
    # cv2.medianBlur(src, ksize)

# 가우시안 블러링
    # 평균 또는 중간값이 아닌 가우시안 분포를 적용한 필터(커널)를 이용한 블러링 방식
    # 경계선과 같은 엣지를 잘 유지하면서 블러링을 처리하는 것이 특징
    # 평균값보다 자연스럽고 부드러운 표현이 가능
    # 가우시안 노이즈에 효과적
    # 가우시안 분포 : 정규 분포라고도 하며, 평균 근처에 모여있는 값들의 개수가 많고 평균에서 멀어질수록 그 개수가 적어지는 분포를 의미함. 표준편차가 커지면 봉오리의 폭이 넓어지면서 낮아지고 표준편차가 작아지면 폭이 좁아지면서 봉오리가 높아진다. 양끝으로 갈수록 수치가 낮아지는 종 모양을 가지고 있으며, 평균을 중심으로 대칭구조를 형성한다.
        # x : 정규 분포에서 확률 밀도(분포)를 계산하려는 변수
        # m : 정규 분포의 평균, 데이터의 중심을 의미.
        # sigma : 정규 분포의 표준 편차. 데이터의 퍼짐 정도.
    # 가우시안 커널의 값들은 합이 항상 1이어야 함.(밝기에 영향을 주지 않기 위해)
    # 가우시안 함수
        # cv2.GaussianBlur(src, ksize, sigmaX [, sigmaY, border])
            # src : 입력 이미지
            # ksize : 커널 크기
            # sigmaX : X 방향 표준편차
                # 0=auto, sigma = 0.3((ksize-1)*0.5-1)+ 0.8이 커널 크기에 따라 시그마를 구하는 공식
            # sigmaY : Y 방향 표준편차, default는 sigmaX
            # border : 테두리 보정 방식

            # kernel = cv2.getGaussianKernel(ksize sigma)
                # ksize : 커널 크기
                # sigma : 표준 편차
                # kernel * kernel.T