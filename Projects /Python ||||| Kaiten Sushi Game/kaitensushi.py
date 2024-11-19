import pygame
import pygame_menu
import random
from pygame.locals import Rect, QUIT, KEYUP, K_SPACE

def run_sushi_game():
    pygame.init()
    pygame.key.set_repeat(5, 5)
    SURFACE = pygame.display.set_mode((1000, 600))
    FPSCLOCK = pygame.time.Clock()

    background = pygame.image.load("image/background.png") # 전체 배경 이미지

    plate = pygame.image.load("image/plate.png") # 접시 이미지
    plate = pygame.transform.scale(plate, (265, 195))

    chairs = pygame.image.load("image/chairs.png") # 의자 이미지
    chairs = pygame.transform.scale(chairs, (250, 400))

    cuttingboard = pygame.image.load("image/cuttingboard.png") # 도마 이미지
    cuttingboard = pygame.transform.scale(cuttingboard, (400, 100))

    sushi_img = pygame.image.load("image/sushi.png") # 초밥 이미지
    sushi_img = pygame.transform.scale(sushi_img, (50, 30))
    bacon = pygame.image.load("image/bacon.png") # 초밥 이미지
    bacon = pygame.transform.scale(bacon, (50, 30))
    meat = pygame.image.load("image/meat.png") # 초밥 이미지
    meat = pygame.transform.scale(meat, (50, 30))
    egg = pygame.image.load("image/egg.png") # 초밥 이미지
    egg = pygame.transform.scale(egg, (50, 30))
    shrimp = pygame.image.load("image/shrimp.png") # 초밥 이미지
    shrimp = pygame.transform.scale(shrimp, (50, 30))

    sushi_img2 = pygame.image.load("image/fish.png") # 죽은 생선 이미지
    sushi_img2 = pygame.transform.scale(sushi_img2, (50,30))

    fish = pygame.image.load("image/fish.png") # 죽은 생선 이미지
    fish = pygame.transform.scale(fish, (200, 100))

    sink = pygame.image.load("image/sink.png") # 싱크대 이미지
    sink = pygame.transform.scale(sink, (310, 130))

    rice = pygame.image.load("image/rice.png") # 밥솥 이미지
    rice = pygame.transform.scale(rice, (150, 150))
    mill = pygame.image.load("image/mill.png") # 밥 이미지
    mill = pygame.transform.scale(mill, (90, 90))

    way = pygame.image.load("image/rail.png") # 레일 이미지
    way = pygame.transform.scale(way, (150, 110))

    carpet = pygame.image.load("image/carpet.png")
    carpet = pygame.transform.scale(carpet, (500,170))
    carpet2 = pygame.image.load("image/carpet2.png")
    carpet2 = pygame.transform.scale(carpet2, (500,170))

    backgroundmusic = pygame.mixer.Sound("sound/backgroundmusic.mp3")
    backgroundmusic.play(-1)
    missionfailmusic  = pygame.mixer.Sound("sound/missionFailed.mp3")
    error = pygame.mixer.Sound("sound/error.mp3")
    hit = pygame.mixer.Sound("sound/hit.mp3")
    eatsound = pygame.mixer.Sound("sound/eat.mp3")
    plateplacing = pygame.mixer.Sound("sound/plateplacing.mp3")
    platebreaking = pygame.mixer.Sound("sound/platebreaking.mp3")

    keyleft = pygame.image.load("image/keyleft.png") # 왼쪽 방향키 이미지
    keyleft = pygame.transform.scale(keyleft, (90, 90))
    keyright = pygame.image.load("image/keyright.png") # 오른쪽 방향키 이미지
    keyright = pygame.transform.scale(keyright, (100, 100))
    keyup = pygame.image.load("image/keyup.png") # 위쪽 방향키 이미지
    keyup = pygame.transform.scale(keyup, (80, 80))
    keydown = pygame.image.load("image/keydown.png") # 아래쪽 방향키 이미지
    keydown = pygame.transform.scale(keydown, (80, 80))

    guy1 = pygame.image.load("image/guy1.png") # 사람 1 이미지
    guy1 = pygame.transform.scale(guy1, (150, 300))

    randomkey = [keyleft,keyright,keyup,keydown]

    onchair = [True, True, True, True] # 의자에 손님이 앉아 있는지 ?
    rails = []
    sushi = []
    road = []
    emptyPlate = []
    Currentplate = 0
    global count
    count = 4
    benefit = 0
    sysfont = pygame.font.Font(None, 100)
    font = pygame.font.Font(None,40)


    for x in range(6): # 접시와 초밥 리스트 초기화
        rect = Rect(x * 200, 270, 150, 50)
        rails.append(rect)
        sushi.append(rect)
        

    for x in range(12): # 레일 리스트 초기화
        rect = Rect(x * 150, 100, 100, 50)
        road.append(rect)

    customer_timer = 0
    customer_interval = 500
    customer_cooldown = 0

    left_rect = Rect(0, 280, 400, 80)
    center_rect = Rect(400, 280, 150, 75)
    right_rect = Rect(550, 280, 600, 80)
    left_plates = [i for i, rail in enumerate(rails) if rail.centerx < center_rect.centerx]
    right_plates = [i for i, rail in enumerate(rails) if rail.centerx > center_rect.centerx]
    center_plates = left_plates + right_plates

    level = 0
    Arrivetime = 0

    sushis = [sushi_img, sushi_img2, bacon, meat, egg,shrimp]
    sushis_copy = [sushi_img, sushi_img2, bacon, meat, egg]

    while True:
        timer = 0
        Arrivetime += 1
        arrowEvent = random.randint(4,4+level) # 방향키 입력 미션이 몇개 생성될 것인지 (4 ~ 7)
        fail = False # 방향키 입력 미션 실패 여부

        SURFACE.fill((255,255,255))
        SURFACE.blit(background,(0,0))

        for x in range(4):  # 의자 이미지 부착
            chair_position = (x * 250 + 30, 150)
            SURFACE.blit(chairs, chair_position, Rect(50, 0, 300, 300))
            
            if onchair[x] == True: # 의자에 손님이 앉아 있을 시 character 이미지를 표시
                SURFACE.blit(guy1, (chair_position[0], chair_position[1]))
            
        for i in range(4): # 의자가 비어 있으면 손님이 착석
            if random.random() < 0.02 and not onchair[i] and Arrivetime >= 400:
                onchair[i] = True
                Arrivetime = 0
                
        pygame.draw.rect(SURFACE, (160, 60, 42), (0, 360, 1000, 600)) # 조리 테이블    
        for i, rail in enumerate(road): # 초밥 이동 레일 부착
            SURFACE.blit(way, (rail.centerx-200, rail.centery + 143))
        # 디자인 - 싱크대
        SURFACE.blit(sink, (50, 350))
        # 디자인 - 초밥 제작 테이블 이미지 부착
        SURFACE.blit(carpet, (-200, 460))
        # 디자인 - 초밥 제작 테이블 이미지 부착
        SURFACE.blit(carpet2, (690, 330))

        pygame.draw.rect(SURFACE, (0, 0, 255), center_rect, 5) # 초밥을 추가 가능한 영역

        # 초밥 이동 레일 이동
        road = [xpos.move(-2, 0) for xpos in road]
        # 초밥 이동 레일이 화면을 벗어나면 다시 오른쪽으로 이동
        if road[0].left <= -20:
            road[0].move_ip(1170, 0)
            road.append(road[0])
            del road[0]

        for i, rail in enumerate(rails):
            SURFACE.blit(plate, (rail.centerx - 138, rail.centery - 98)) # 접시 이미지 부착
            sushi_position = (rail.centerx - 25, rail.centery - 25)
            current_sushi = sushis[i % len(sushis)]  # 초밥 종류 순환 #수정필요
            print(current_sushi)
            SURFACE.blit(current_sushi, sushi_position)  # 초밥 이미지 부착
            # 손님이 의자에 앉아 있는 상태에서 충돌 확인
            if onchair[i // 2] and pygame.Rect(sushi_position, (50, 50)).colliderect(pygame.Rect((i * 250 + 30, 300), (100, 300))):
            # 손님이 의자에 앉아 있을 때, 약 10%확률로 접시를 회수
                r = random.randint(1, 1000) #수정필요
                if r < 20:
                    eatsound.play()
                    Currentplate += 1 # 싱크대에 접시를 쌓기 위한 카운트
                    del rails[i]
                    del sushi[i]
                    sushis.append(sushis[i])
                    del sushis[i]
                    benefit += 1000 # 손님이 초밥을 가져다 먹으면 이익 발생

        for i in range(Currentplate): # 싱크대에 접시를 하나 쌓음
            SURFACE.blit(plate, (70, 350 - i * 10))

        SURFACE.blit(cuttingboard, (300, 370)) # 도마 이미지 부착
        SURFACE.blit(rice, (650, 300)) # 밥솥 이미지 부착
        SURFACE.blit(mill, (630, 385)) # 밥 이미지 부착
        SURFACE.blit(fish, (800, 360)) # 죽은 생선 이미지 부착

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == KEYUP:
                if event.key == pygame.K_UP:
                    if any(rail.colliderect(center_rect) for rail in rails): # 중앙 파란 테두리와 레일 간의 충돌 확인
                        error.play()
                        print("추가할 수 없습니다. 이미 초밥이 있습니다.")
                    else:
                        time = 0 # 미션을 수행하는 동안 줄어들 게이지의 시간 카운트
                        key_images = [random.choice(randomkey) for _ in range(arrowEvent)] # 미션의 방향키는 랜덤으로 결정함.
                        while arrowEvent > 0: # 미션 수행을 완벽하게 완료하기 전까지,
                            time += 1 # 시간은 1씩 증가
                            if(time < 2200): # 타이머가 2000에 도달하기 전까지,
                                pygame.draw.rect(SURFACE, (255, 255, 0), (int(time / 4), 570,int(time / 4), 20)) # 노란색 게이지를 줄여나감.
                            else: # 2000에 도달하면,
                                platebreaking.play()
                                benefit -= 2000
                                break # 미션 실패
                            pygame.draw.rect(SURFACE, (160, 60, 42), (290, 470, 800, 80)) # 방향키 이미지가 이상하게 출력되는 문제를 해결하기 위해 배경색을 덧칠
                            for i in range(0, arrowEvent): # 랜덤으로 설정된 방향키 미션 이미지 부착.
                                SURFACE.blit(key_images[i], (300 + i * 90, 470))
                                
                            pygame.display.update()

                            # 키 입력 확인
                            for event_keyup in pygame.event.get():
                                pygame.display.update()
                                if event_keyup.type == KEYUP:
                                    # 방향키 입력 미션을 정확하게 수행하고 있는지 판단.
                                    if event_keyup.key == pygame.K_LEFT and key_images[0] == keyleft:
                                        hit.play()
                                        print("왼쪽 방향키 눌림")
                                        del key_images[0] # 올바르게 누르면 맨 앞 이미지를 삭제하고 다음 순서의 이미지를 앞으로 당겨옴.
                                        arrowEvent -= 1 # 남은 방향키의 수 하나 감소시킴.
                                    elif event_keyup.key == pygame.K_RIGHT and key_images[0] == keyright:
                                        print("오른쪽 방향키 눌림")
                                        hit.play()
                                        del key_images[0]
                                        arrowEvent -= 1
                                    elif event_keyup.key == pygame.K_UP and key_images[0] == keyup:
                                        print("위쪽 방향키 눌림")
                                        hit.play()
                                        del key_images[0]
                                        arrowEvent -= 1
                                    elif event_keyup.key == pygame.K_DOWN and key_images[0] == keydown:
                                        print("아래쪽 방향키 눌림")
                                        hit.play()
                                        del key_images[0]
                                        arrowEvent -= 1
                                    else: # 방향키를 잘못 입력했을 경우,
                                        print("와장창 !")
                                        platebreaking.play()
                                        if arrowEvent == 4:
                                            benefit -= 1500
                                        elif arrowEvent == 5:
                                            benefit -= 1700
                                        elif arrowEvent == 6:
                                            benefit -= 1800
                                        elif arrowEvent == 7:
                                            benefit -= 2000
                                        fail = True # 미션 실패 플래그 ON
                                        break # 방향키 미션을 강제로 빠져나감.

                            if fail: # 미션 실패 플래그가 켜져 있을 시,
                                del(key_images) # 현재 리스트에 들어있는 모든 방향키 이미지를 삭제시킴. 정확히는 리스트 자체를 삭제.
                                break

                            if arrowEvent == 0: # 모든 방향키 입력을 성공적으로 마치면,
                                new_sushi_rect = Rect(center_rect.centerx - 25, center_rect.centery - 25, 50, 50) # 레일에 추가할 초밥과 접시의 객체.
                                rails.insert(len(left_plates)+1,Rect(400, 260, 150, 70)) # 모든 초밥들의 인덱스 중 정가운데 인덱스에 끼워넣음.
                                Currentplate -= 1 # 싱크대에서 접시를 하나 가져와서
                                sushi.insert(len(left_plates)+1, new_sushi_rect) # 초밥을 담아 레일에 올림.
                                sushis.insert(len(left_plates)+1, random.choice(sushis_copy))
                                plateplacing.play()
                                print("추가")

        if benefit < 10000:
            level = 1
        elif benefit > 10000:
            level = 2
        elif benefit > 20000:
            level = 3
        elif benefit > 30000:
            level = 4
            
        if benefit >= 40000: # 매출이 40000원을 넘어가면 게임 클리어 !
            clear = sysfont.render("Clear !!!", True, (0, 0, 255))
            pygame.display.set_caption('Clear !!!')
            SURFACE.blit(clear, (500,300))
            pygame.display.update()
            pygame.time.wait(3000)
            break

        if len(sushi) == 0 or len(rails) == 0 or benefit < 0: # 레일 위에 접시가 모두 소진되거나, 이익이 0원 이하로 내려갈 시 게임 오버.
            game_over = sysfont.render("Game Over", True, (255, 0, 0))
            SURFACE.blit(game_over, (500,300))
            pygame.display.set_caption('Game Over')
            backgroundmusic.stop()
            missionfailmusic.play()
            pygame.display.update()
            pygame.time.wait(3000)
            break


        if rails[0].centerx <= -70 or sushi[0].centerx <= -70: # 접시가 왼쪽 벽을 넘어가면 오른쪽에서 생성되어 다시 나오게 함.
            sushis.append(sushis[0])
            sushis[0],sushis[1] = sushis[1],sushis[0]
            del sushis[1]
            rails.append(rails[0])
            sushi.append(sushi[0])
            rails[0].move_ip(1300, 0)
            sushi[0].move_ip(1300, 0)
            print(current_sushi)
            del rails[0]
            del sushi[0]

        rails = [xpos.move(-2, 0) for xpos in rails] # 레일의 x좌표를 -2씩 이동시킴.
        sushi = [xpos.move(-2, 0) for xpos in sushi] # 초밥의 x좌표를 -2씩 이동시킴.

        currentMoney = font.render("Current Benefit : {}".format(benefit), True, (255,0,0))
        SURFACE.blit(currentMoney, (0,0)) # 좌측 상단에 게임의 점수를 표시

        pygame.display.update()
        FPSCLOCK.tick(25)

        customer_timer += 1 # 고객 타이머 업데이트
        customer_interver = (100,400)
        if customer_timer >= customer_interval: # 새로운 고객이 도착할 시간인지를 판단.
            leaving_chair = random.choice(range(4)) # 고객이 떠날 의자를 선택.
            onchair[leaving_chair] = False # 고객이 떠난 의자는 비었음을 표시.

            # 타이머 초기화하고 다음 고객 도착 간격을 랜덤하게 설정
            customer_timer = 0

pygame.init()
SURFACE = pygame.display.set_mode((1000, 600))
pygame.display.set_caption('Sushi Game Menu')

menu = pygame_menu.Menu('Menu', 1000, 600)
menu.add.button('Start Sushi Game', run_sushi_game)
menu.add.button('Quit', pygame_menu.events.EXIT)
menu.mainloop(SURFACE)
pygame.quit()

