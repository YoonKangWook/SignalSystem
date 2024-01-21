import numpy as np
import ConvCodec as cc
import matplotlib.pyplot as plt

data_size = 200 # 200개의 data bit 생성
max_snr = 13
ber = []
recurring_size = 10 # 반복 횟수


for snr_db in range(0, max_snr):
    bit_error_count = 0
    for _ in range(0, recurring_size): # 200개의 데이터를 5000번 반복 -> snr마다 1000000번 돌리고 결과 누적
        data = np.random.randint(0, 2, data_size) # 0 or 1 data 생성 (200, )
        print(data)
        print("=============")
        encoded_bit = cc.Encoder(data)            # (2, 1024 + 3(0, 0, 0 tail bit)), 0과 1로 구성

        real_signal = encoded_bit[0, :] * 2 - 1   # -1과 1을 발생
        imag_signal = encoded_bit[1, :] * 2 - 1   # -1과 1을 발생
        
        qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
        # IFFT 블록
        ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size) # 줄어든 파워만큼 보상, 평균 파워가 1이 되도록
        # ofdm_sym = np.fft.ifft(qpsk_sym) #QPSK 심볼 에너지가 data_size 배 감소한다.

        noise_std = 10 ** (-snr_db / 20)
        noise = np.random.randn(data_size+3) * noise_std / np.sqrt(2) + 1j * np.random.randn(data_size+3) * noise_std / np.sqrt(2)
        rcv_signal = ofdm_sym + noise
        rcv_signal = np.fft.fft(rcv_signal) / np.sqrt(data_size)

        real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size+3)
        imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size+3)

        # real_detected_signal[0][5:5] = (real_detected_signal[0][5:5] + 1) % 2
        # imag_detected_signal[0][5:5] = (imag_detected_signal[0][5:5] + 1) % 2
        #
        # real_detected_signal[0][10:11] = (real_detected_signal[0][10:11] + 1) % 2
        # imag_detected_signal[0][10:11] = (imag_detected_signal[0][10:11] + 1) % 2


        print(type(real_detected_signal))
        # decoder로 넣기 위해 데이터 가공 (2, data_size + 3)개로
        dec_input = np.vstack([real_detected_signal, imag_detected_signal])
        # print(np.shape(dec_input))
        decoded_bit = cc.ViterbiDecoder(dec_input)
        print(type(decoded_bit))
        print(decoded_bit)
        print("=============")

        # 비교 출력
        # print(np.sum(np.abs(dec_input - encoded_bit))) # 오류 정정 전 결과
        # print(np.sum(np.abs(data - decoded_bit)))      # 오류 정정 후 결과

        # print(data)
        # print(decoded_bit)

        # 에러 수
        bit_error_count += np.sum(np.abs(dec_input - encoded_bit))
    tmp_ber = bit_error_count / (data_size * 2 * 5000)
    ber.append(tmp_ber)

snr = np.arange(0, max_snr)
plt.semilogy(snr, ber)
plt.show()


