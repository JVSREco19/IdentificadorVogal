import librosa
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

def acquire_signal(sampling_rate, duration):
  # Adquirir sinal do microfone usando sounddevice
  print(f"Aquisição de sinal em andamento (taxa de amostragem: {sampling_rate} Hz, duração: {duration} segundos)...")
  signal = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype='float64')
  sd.wait()  # Aguardar o término da aquisição
  signal = signal.flatten()  # Converter para array unidimensional
  print("Aquisição concluída.")
  return signal

def plot_signal_and_spectrum(signal, sampling_rate):
  # Verificar se o comprimento do sinal é válido
  if len(signal) == 0:
    print("Erro: Comprimento do sinal é zero.")
    return

  # Calcular a FFT do sinal
  spectrum = np.fft.fft(signal)
  frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)

  # Plotar o sinal no domínio do tempo
  plt.subplot(2, 1, 1)
  plt.plot(np.arange(len(signal)) / sampling_rate, signal)
  plt.title('Sinal no Domínio do Tempo')
  plt.xlabel('Tempo (s)')
  plt.ylabel('Amplitude')

  # Plotar o espectro no domínio da frequência
  plt.subplot(2, 1, 2)
  plt.plot(frequencies, np.abs(spectrum))
  plt.title('Espectro de Frequência')
  plt.xlabel('Frequência (Hz)')
  plt.ylabel('Magnitude')

  plt.tight_layout()
  plt.show()
  

def plot_mfccs(signal, sampling_rate):
  mfccs = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13)
  
  # Plotar os coeficientes cepstrais de frequência mel (MFCCs)
  plt.figure(figsize=(10, 4))
  librosa.display.specshow(mfccs, x_axis='time')
  plt.colorbar()
  plt.title('MFCCs')
  plt.show()
  
  
def load_and_plot_audio(file_path, sampling_rate, duration):
  # Carregar arquivo de áudio usando librosa
  audio_data, original_sampling_rate = librosa.load(file_path, sr=None)
  # audio_data2 = acquire_signal(sampling_rate,duration)
  
  
  # Verificar se o comprimento do sinal é válido antes de calcular a FFT
  plot_signal_and_spectrum(audio_data, sampling_rate)
  
  # # Verificar se o comprimento do sinal é válido antes de calcular a FFT
  # plot_signal_and_spectrum(audio_data2, sampling_rate)
  
  plot_mfccs(audio_data,sampling_rate)

# Configurar parâmetros de aquisição
file_path = 'vogais.wav'
sampling_rate = 44100  # Exemplo: taxa de amostragem de 44.1 kHz
duration = 5  # Exemplo: duração de 5 segundos

# Carregar e plotar o áudio com os parâmetros configurados
load_and_plot_audio(file_path, sampling_rate, duration)
