import numpy as np

# K = 3 Codec 구현

def Encoder(data):
    data = np.append(data, [0, 0, 0]) # data + Tail bit 설정 => Input data 초기화
    dataSize = np.shape(data)[0] # 1행으로 된 데이터 ex) (64, )
    shiftReg = [0, 0, 0] # K = 3, Shift Register 설정 및 초기화
    encoded_bit = np.zeros((2, dataSize)) # R = 1 / 2, 1비트 들어오면 2비트 출력 => 2행 dataSize(열) 생성
    # Input data 이동 과정
    for i in range(dataSize):
        shiftReg[2] = shiftReg[1]
        shiftReg[1] = shiftReg[0]
        shiftReg[0] = data[i]
        # 위로 나가는 출력
        encoded_bit[0, i] = np.logical_xor( np.logical_xor(shiftReg[0], shiftReg[1]), shiftReg[2] )
        # 아래로 나가는 출력
        encoded_bit[1, i] = np.logical_xor(shiftReg[0], shiftReg[2])

    return encoded_bit


def ViterbiDecoder(encoded_bit):
    ref_out = np.zeros((2, 8)) # 4개의 state로 들어오는 8개의 output
    ref_out[0, :] = [0, 1, 1, 0, 1, 0, 0, 1]
    ref_out[1, :] = [0, 1, 0, 1, 1, 0, 1, 0]
    # (00 / 01 / 10 / 11) 각각의 state로 들어오는 화살표의 출력 값들
    # 00으로 들어오는 것의 과거 state는 00과 01 출력 값은 00 11
    # 11으로 들어오는 것의 과거 state는 10과 11 출력 값은 01 10

    dataSize = np.shape(encoded_bit)[1] # 2 by ( 원래 데이터 길이 + 3 [0, 0, 0] ) = encoded_bit의 dataSize
    cumDist = [0, 100, 100, 100] # 누적 거리값 설정 00 / 01 / 10 / 11
    prevState = [] # 이전 state 경로 기록

    # print(encoded_bit)
    # 수신
    # [[1. 1. 0. 1. 1. 0. 0.]
    #  [1. 0. 0. 0. 1. 0. 0.]]
    # print(dataSize) == 7

    for i in range(0, dataSize): # 비트 결과 비교
        tmpData = np.tile(encoded_bit[:, i].reshape(2, 1), (1, 8)) # tile -> 붙이는 것
        # encoded_bit -> 11 10 00 11
        # 1 -> 1 1 1 1 1 1 1 1
        # 1 -> 1 1 1 1 1 1 1 1
        # 복사하는 이유 output과 거리를 비교하기 위해서
        dist = np.sum(np.abs(tmpData - ref_out), axis=0) # 0 ~ 2사이에 8개의 값이 나옴
        # -1  0  0 -1  0 -1 -1  0
        # -1  0 -1  0  0 -1  0 -1
        #  2  0  1  1  0  2  1  1 # dist 계산 결과
        tmpDist = np.tile(cumDist, (1, 2)) + dist # 누적 거리를 1행으로 2개 복사해서 dist값 더하기

        # 검증 코드
        # print(tmpData)
        # print("@@@@@@@@@@@@@@@@@@@@@@@@")
        # print(dist)
        # print("#############################")
        # print(np.tile(cumDist, (1, 2)))
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(tmpDist)
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(dist)
        # [0 100 100 100 0 100 100 100] + [2. 0. 1. 1. 0. 2. 1. 1.] (dist 결과)
        tmpPrevState = []  # 과거의 state 기록
        for a in range(4): # state 수 = 4개
            if tmpDist[0, 2 * a + 0] <= tmpDist[0, 2 * a + 1]: # 누적 거리가 작은 값 선택 과정, 2개마다 비교
                cumDist[a] = tmpDist[0, 2 * a + 0]
                tmpPrevState.append((a % 2) * 2 + 0) # state 위치를 나타냄 00 / 01 / 10 / 11
            else:
                cumDist[a] = tmpDist[0, 2 * a + 1]
                tmpPrevState.append((a % 2) * 2 + 1)
        prevState.append(tmpPrevState)
        # print(tmpPrevState) # 8개 값을 2개씩 비교
        # print(cumDist)
        state_index = np.argmin(cumDist) # 마지막에서 누적 거리가 가장 짧은 것의 인덱스 찾기
        # print(state_index, cumDist) # 결과 (도착 인덱스, cD값)
        # 2 [2.0, 101.0, 0.0, 101.0]
        # 1 [3.0, 0.0, 3.0, 2.0]
        # 0 [0.0, 3.0, 2.0, 3.0]
        # 0 [0.0, 3.0, 2.0, 3.0]
        # 00 01 10 11 / cD : 0 2 3 2 -> index = 0
        # 00, 01 -> 입력이 0 / 10, 11 -> 입력이 1
    # print(prevState)

    # Decoding
    decoded_bit = []
    for b in range(dataSize - 1, -1, -1): # 디코딩 과정은 역순부터
        decoded_bit.append(int(state_index / 2)) # 인덱스 0, 1은 입력 0, 인덱스 1, 2는 입력 1
        state_index = prevState[b][state_index]
        # print(state_index)
        # print(prevState)
    data_size = np.shape(decoded_bit)[0]
    decoded_bit = np.flip(decoded_bit)[0 : data_size - 3] # 순서를 뒤짚고 실제 정보 비트만 저장

    return decoded_bit

# print(ViterbiDecoder(Encoder([1, 0, 1, 0])))
# ViterbiDecoder(Encoder([1, 0, 1, 0]))