import numpy as np

# interleaver

input_bit = [0 for _ in range(4)]

# 4개의 심볼 입력, 16비트
def interleaver(data):
    print(data)
    for i in range(4):
        for j in range(4):
            input_bit[i][j] = data[ 4 * i + j]



    # for i in range(int(len(data[0]))):
    #     input_bit.append(data[:, i])

    print(input_bit)
    return input_bit

# 0 ~ 15
list = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1])

interleaver(list)
# interleaver(list.reshape(4, 4))