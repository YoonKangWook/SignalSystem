import matplotlib.pyplot as plt
import numpy as np
import ConvCodec_k4 as cc_k4

data_size=128   #하나의 OFDM 심볼에 한번에 전송되는 QPSK 심볼의 수
max_snr=13      #최대 SNR 13db까지 실험
max_repeat = 1000 # 반복 횟수
ber=[] # channel coding 적용
Viterbi_ber=[]


for snr_db in range(0, max_snr):
        total_error = 0
        viterbi_total_error = 0
        for _ in range(0, max_repeat):  # 200개의 데이터를 5000번 반복 -> snr마다 1000000번 돌리고 결과 누적
                data = np.random.randint(0, 2, data_size)  # 0 or 1 data 생성 (200, )
                encoded_bit = cc_k4.Encoder_k4(data)  # (2, 1024 + 3(0, 0, 0 tail bit)), 0과 1로 구성
                # print(encoded_bit[0,:])

                encoded_bit[0, 100:110] = (encoded_bit[0, 100:110] + 1) % 2
                encoded_bit[1, 100:110] = (encoded_bit[1, 100:110] + 1) % 2

                # # 100~110번째 비트 바꾸기
                # encoded_bit[0, 100:100] = (encoded_bit[0, 100:100] + 1) % 2
                # encoded_bit[1, 100:100] = (encoded_bit[1, 100:100] + 1) % 2
                # encoded_bit[0, 105:105] = (encoded_bit[0, 105:105] + 1) % 2
                # encoded_bit[1, 105:105] = (encoded_bit[1, 105:105] + 1) % 2
                # encoded_bit[0, 108:108] = (encoded_bit[0, 108:108] + 1) % 2
                # encoded_bit[1, 108:108] = (encoded_bit[1, 108:108] + 1) % 2
                # encoded_bit[0, 50:50] = (encoded_bit[0, 50:50] + 1) % 2
                # encoded_bit[1, 50:50] = (encoded_bit[1, 50:50] + 1) % 2

                # print(encoded_bit[0])
                real_signal = encoded_bit[0, :] * 2 - 1  # -1과 1을 발생
                imag_signal = encoded_bit[1, :] * 2 - 1  # -1과 1을 발생

                # 인터리버

                qpsk_sym = (real_signal + 1j * imag_signal) / np.sqrt(2)
                # IFFT 블록
                ofdm_sym = np.fft.ifft(qpsk_sym) * np.sqrt(data_size)  # 줄어든 파워만큼 보상, 평균 파워가 1이 되도록
                # print(np.sum(np.abs(ofdm_sym) ** 2) / data_size) #QPSK 심볼 에너지가 data_size 배 감소한다.

                noise_std = 10 ** (-snr_db / 20)
                noise = np.random.randn(data_size + 4) * noise_std / np.sqrt(2) + 1j * np.random.randn(
                        data_size + 4) * noise_std / np.sqrt(2)
                rcv_signal = ofdm_sym + noise
                rcv_signal = np.fft.fft(rcv_signal) / np.sqrt(data_size)
                # print(np.sum(np.abs(rcv_signal) ** 2) / data_size)

                real_detected_signal = np.array(((rcv_signal.real > 0) + 0)).reshape(1, data_size + 4)
                imag_detected_signal = np.array(((rcv_signal.imag > 0) + 0)).reshape(1, data_size + 4)
                # print(np.shape(real_detected_signal))
                # print(np.shape(imag_detected_signal))

                # print("================")
                # print(real_detected_signal)
                # print("================")
                # print(imag_detected_signal)


                # decoder로 넣기 위해 데이터 가공 (2, data_size + 4)개로
                dec_input = np.vstack([real_detected_signal, imag_detected_signal])
                # print(np.shape(dec_input))
                # print("================")
                # print(dec_input)
                # print("================")


                # 디-인터리버

                decoded_bit = cc_k4.ViterbiDecoder_k4(dec_input)

                # 비교 출력
                # print(np.sum(np.abs(dec_input - encoded_bit))) # 오류 정정 전 결과
                # print(np.sum(np.abs(data - decoded_bit)))      # 오류 정정 후 결과

                # 에러 수
                total_error += np.sum(np.abs(dec_input - encoded_bit))
                viterbi_total_error += np.sum(np.abs(data - decoded_bit))

        tmp_ber = total_error / (data_size * 2 * max_repeat)
        viterbi_ber = viterbi_total_error / (data_size * 2 * max_repeat)


        ber.append(tmp_ber)
        Viterbi_ber.append(viterbi_ber)



# 나중에 할 것
# 2) 인터리버 구현

# 1) 채널코딩 적용안한 QPSK ber
bit_size=10000*128
nch_ber = []
for snr_db in range(0, max_snr):
    # -1 or 1 출력
    i_data=np.random.randint(0,2,bit_size)*2-1 # 실수 부분 cos
    q_data=np.random.randint(0,2,bit_size)*2-1 # 허수 부분 sin

    # 심볼 수 n개
    sym=(i_data+1j*q_data)/np.sqrt(2)

    # OFDM 블록
    ofdm = np.fft.ifft(sym)
    # print(np.sum(np.abs(ofdm) ** 2) / bit_size) # 에너지는 똑같이 유지되야함 -> 결과: 0.0039062 -> 보상이 필요함
    ofdm = ofdm * np.sqrt(bit_size)  # 보상
    # print(np.sum(np.abs(ofdm) ** 2) / bit_size)

    noise_std = 10 ** (-snr_db / 20)  # 노이즈의 표준편차
    noise = (np.random.randn(bit_size) + 1j * np.random.randn(bit_size)) / np.sqrt(2)  # N_sc만큼의 개수를 생성, 실수와 허수로 갈라지기 때문에 루트 2로 나눔
    noise = noise * noise_std
    # 신호 수신
    rcv_sig = ofdm + noise

    # FFT 블록
    rcv_sig = np.fft.fft(rcv_sig)  # 에너지가 동일하다는 보장이 없음 -> 보상 필요
    # print(np.sum(np.abs(rcv_sig) ** 2 ) / bit_size)
    rcv_sig = rcv_sig / np.sqrt(bit_size)  # 보상 과정
    # print(np.sum(np.abs(rcv_sig) ** 2) / bit_size)
    # IFFT 할 때는 곱해주고 FFT할 때는 나줘줘야함 -> 파이썬 기준


    # 1사분면 신호 판정
    i_dect=(rcv_sig.real>0)*2-1
    q_dect=(rcv_sig.imag>0)*2-1

    # 실수 에러 개수
    i_error = np.sum(np.abs(i_data - i_dect) / 2)

    # 허수 에러 개수
    q_error = np.sum(np.abs(q_data - q_dect) / 2)

    # BER
    error_rate = (i_error + q_error) / (bit_size * 2)

    nch_ber.append(error_rate)


snr = np.arange(0, max_snr)

plt.title('OFDM_sys_ber & N_ch_ber')
plt.semilogy(snr, ber, label='OFDM_sys_ber', color = 'blue')
plt.semilogy(snr, Viterbi_ber, label='Viterbi_ber', color = 'violet')
plt.semilogy(snr, nch_ber, label='N_ch_ber', color = 'red')
plt.legend()
plt.xlabel('SNR')
plt.ylabel('BER')

plt.show()
