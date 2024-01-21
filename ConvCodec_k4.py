import numpy as np

# K = 4 Codec 구현

def Encoder_k4(data):
    data = np.append(data, [0, 0, 0, 0]) # data + tail bit(4개)
    dataSize = np.shape(data)[0] # 데이터 길이 (128 + 4, )
    shiftReg = [0, 0, 0, 0] # K = 4, Shift Register 초기화
    encoded_bit = np.zeros((2, dataSize)) # R = 1/2,

    # Input data 이동 과정
    for i in range(dataSize):
        shiftReg[3] = shiftReg[2]
        shiftReg[2] = shiftReg[1]
        shiftReg[1] = shiftReg[0]
        shiftReg[0] = data[i]

        # 위로 나가는 출력
        encoded_bit[0, i] = np.logical_xor(np.logical_xor(np.logical_xor(shiftReg[0], shiftReg[1]), shiftReg[2]), shiftReg[3])
        # 아래로 나가는 출력
        encoded_bit[1, i] = np.logical_xor(np.logical_xor(shiftReg[0], shiftReg[2]), shiftReg[3])

    return encoded_bit


# print(np.shape(Encoder_k4([1, 0, 1, 0])))


def ViterbiDecoder_k4(encoded_bit):
    ref_out = np.zeros((2, 16)) # 8개의 state로 들어오는 16개의 output
    ref_out[0, :] = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    ref_out[1, :] = [0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1]
    # (000 / 001 / 010 / 011 / 100 / 101 / 110 / 111)의 state로 들어오는 화살표의 출력 값들

    dataSize = np.shape(encoded_bit)[1] # 2 by (원래 데이터 길이 + 4) = encoded_bit의 dataSize
    cumDist = [0, 100, 100, 100, 100, 100, 100, 100] # 누적 거리값 설정
    prevState = [] # 이전 state 경로 기록

    for i in range(0, dataSize): # 비트 결과 비교
        tmpData = np.tile(encoded_bit[:, i].reshape(2, 1), (1, 16))
        # encoded_bit -> 11 10 00 01
        # 1 -> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        # 1 -> 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
        # output과 state의 출력 값의 거리 비교
        dist = np.sum(np.abs(tmpData - ref_out), axis=0)
        # -1  0  0  -1   0  -1  -1   0  0  -1  -1  0  -1  0  0  -1
        # -1  0  0  -1  -1   0   0  -1  0  -1  -1  0  0  -1  -1  0
        #  2  0  0  2   1    1   1   1  0   2   2  0  1   1   1   1
        tmpDist = np.tile(cumDist, (1, 2)) + dist # 누적거리를 2개 복사해서 dist값 더하기

        tmpPrevState = [] # 과거의 state 기록
        for a in range(8): # state 수 = 8개
            if tmpDist[0, 2 * a + 0] <= tmpDist[0, 2 * a + 1]:
                cumDist[a] = tmpDist[0, 2 * a + 0]
                tmpPrevState.append((a % 4) * 2 + 0) # state 위치를 나타냄
            else:
                cumDist[a] = tmpDist[0, 2 * a + 1]
                tmpPrevState.append((a % 4) * 2 + 1)
        prevState.append(tmpPrevState)

        state_index = np.argmin(cumDist) # 마지막에서 누적거리가 가장 짧은 것의 인덱스 찾기
        # print(state_index, cumDist) # 결과 (도착 인덱스, cD값)

    # Decoding
    decoded_bit = []
    for b in range(dataSize-1, -1, -1): # 디코딩 과정 역순부터
        decoded_bit.append(int(state_index / 4)) # 인덱스 0, 1, 2, 3은 입력 0 / 인덱스 4, 5, 6, 7은 입력 1
        state_index = prevState[b][state_index] ########

    data_size = np.shape(decoded_bit)[0]
    decoded_bit = np.flip(decoded_bit)[0 : data_size - 4] # 순서를 뒤짚고 실제 정보 비트만 저장

    return decoded_bit

























