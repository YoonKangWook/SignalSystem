import matplotlib.pyplot as plt
import numpy as np
import ConvCodec as cc

# 주의사항 Viterbi Decoder에는 입력 데이터의 개수를 제한시켜야 한다.
# Viterbi Decoder의 input_bit의 개수는 200개로 제한
# 200개의 bit마다 인코딩 디코딩 다시 수행

# data_size = 100 # 100개의 data bit 생성 -> 인코딩 시 (2, 103)의 데이터 생성 -> 총 206개의 bit 생성(tail bit 3개씩 실수부 허수부에 추가됨)
# recurring_size = 10000 # 반복 횟수
data_size = 200
recurring_size = 5000 # 반복 횟수
max_snr = 11
snr = np.arange(0, max_snr)
rec_size = np.arange(0, recurring_size)
encBer = []
decBer = []

# Convolutional Codec 적용
for snr_db in snr:
    pre_error_count = 0
    cor_error_count = 0
    for _ in rec_size:
        bit_data = np.random.randint(0, 2, data_size)

        # 인코딩, 결과 2행의 0, 1 비트 발생
        encoded_bit = cc.Encoder(bit_data)
        # print(encoded_bit)
        # print(np.shape(encoded_bit))

        real_signal = encoded_bit[0, :] * 2 - 1  # -1과 1을 발생
        imag_signal = encoded_bit[1, :] * 2 - 1  # -1과 1을 발생

        qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
        # print(np.abs(qpsk_sym) ** 2) # 에너지 1 확인

        # OFDM 블록
        trans_ofdm = np.fft.ifft(qpsk_sym)
        # print(np.sum(np.abs(trans_ofdm) ** 2) / data_size) # 에너지는 똑같이 유지되야함 -> 결과: 0.0039062 -> 보상이 필요함
        rcv_ofdm = trans_ofdm * np.sqrt(data_size)  # 보상
        # print(np.sum(np.abs(rcv_ofdm) ** 2) / data_size)

        noise_std = 10 ** (-snr_db / 20)  # 노이즈의 표준편차
        noise = (np.random.randn(data_size + 3) + 1j * np.random.randn(data_size + 3)) / np.sqrt(2)  # data_size + 3(bit + tail_bit(3)만큼의 개수를 생성, 실수와 허수로 갈라지기 때문에 루트 2로 나눔
        noise = noise * noise_std

        # 신호 수신
        rcv_sig = rcv_ofdm + noise

        # FFT 블록
        rcv_signal = np.fft.fft(rcv_sig)  # 에너지가 동일하다는 보장이 없음 -> 보상 필요
        # print(np.sum(np.abs(rcv_signal) ** 2 ) / data_size)
        rcv_signal = rcv_signal / np.sqrt(data_size)  # 보상 과정
        # print(np.sum(np.abs(rcv_signal) ** 2) / data_size)
        # IFFT 할 때는 곱해주고 FFT할 때는 나줘줘야함 -> 파이썬 기준

        # -1 + 0 = 0 / 1 + 0 = 1 : -1, 1 판단 후 변환 과정, 디코딩하기 위해 0 1로 변환
        real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size + 3)
        imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size + 3)

        # decoder로 넣기 위해 데이터 가공 (2, data_size + 3)개로
        dec_input = np.vstack([real_detected_signal, imag_detected_signal])
        # print(encoded_bit)
        # print(np.shape(dec_input))
        decoded_bit = cc.ViterbiDecoder(dec_input)
        # print(decoded_bit)

        # 비교 출력
        # print(np.sum(np.abs(dec_input - encoded_bit)))    # 오류 정정 전 결과
        # print(np.sum(np.abs(bit_data - decoded_bit)))     # 오류 정정 후 결과

        # 에러 개수 결과 저장
        pre_error_count += np.sum(np.abs(dec_input - encoded_bit)) # 정정 전 결과 저장
        cor_error_count += np.sum(np.abs(bit_data - decoded_bit))      # 정정 후 결과 저장

    # 에러율
    enc_ber = pre_error_count / (data_size * 2 * recurring_size) # ex) 200개의 심볼에서 실수부 허수부 비트 총 400개 생성 * 반복횟수
    dec_ber = cor_error_count / (data_size * 2 * recurring_size)

    # 배열에 추가
    encBer.append(enc_ber)
    decBer.append(dec_ber)

    print("snr_db : ", snr_db)
    print(snr_db, " enc_ber : ", enc_ber)
    print(snr_db, " dec_ber : ", dec_ber)


# 그래프 출력
plt.semilogy(snr, decBer, color='blue', label='enc_ber' )
plt.semilogy(snr, encBer, color='red', label='dec_ber')
plt.legend(loc=1)
plt.show()

    


