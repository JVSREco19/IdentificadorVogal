import tkinter as tk
from tkinter import filedialog
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import flattop
from scipy.io.wavfile import write
import sounddevice as sd

class AudioAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer App")

        # Variável para armazenar o áudio gravado
        self.audio_data = None
        self.recorded_audio = False

        # Variáveis para armazenar os parâmetros
        self.sampling_rate_var = tk.StringVar(value="44100")
        self.duration_var = tk.DoubleVar(value=5.0)
        self.window_size_var = tk.StringVar(value="180000")
        self.window_position_var = tk.StringVar(value="0")
        self.selected_window_type = tk.StringVar(value="Nenhum")  

        # Tipos de janelas disponíveis
        self.window_types = ["Nenhum", "hamming", "hanning", "rectangular", "triangular", "flat_top"]

        # Criar widgets
        self.create_widgets()

    def create_widgets(self):
        # Botão para carregar arquivo de áudio
        tk.Button(self.root, text="Carregar Áudio", command=self.load_audio).pack(pady=10)

        # Botão para gravar áudio
        tk.Button(self.root, text="Gravar Áudio", command=self.record_audio).pack(pady=10)
        
        # Botão para utilizar o áudio gravado
        tk.Button(self.root, text="Utilizar Áudio Gravado", command=self.process_audio).pack(pady=10)

        # Configurações de aquisição de sinal
        tk.Label(self.root, text="Configurações de Aquisição de Sinal").pack(pady=5)
        tk.Label(self.root, text="Taxa de Aquisição (Hz):").pack()
        tk.Entry(self.root, textvariable=self.sampling_rate_var).pack()

        # Controle deslizante para o tempo de aquisição
        tk.Label(self.root, text="Tempo de Aquisição (s):").pack()
        tk.Scale(self.root, from_=1, to=10, resolution=0.1, orient="horizontal", variable=self.duration_var).pack()

        # Configurações de cálculo de TDF
        tk.Label(self.root, text="Configurações de Cálculo de TDF").pack(pady=5)
        tk.Label(self.root, text="Tamanho da Janela (amostras):").pack()
        tk.Entry(self.root, textvariable=self.window_size_var).pack()
        tk.Label(self.root, text="Ajuste de Posição da Janela (s):").pack()
        tk.Entry(self.root, textvariable=self.window_position_var).pack()

        # Menu de seleção para o tipo de janela
        tk.Label(self.root, text="Escolher Tipo de Janela").pack(pady=5)
        window_type_menu = tk.OptionMenu(self.root, self.selected_window_type, *self.window_types)
        window_type_menu.pack()
        # Botão para rodar a função plot_mfccs
        tk.Button(self.root, text="Plotar MFCCs", command=self.plot_mfccs).pack(pady=10)

        # Botão para plotar o gráfico
        tk.Button(self.root, text="Plotar Gráfico", command=self.plot_graph).pack(pady=10)
   
    def process_audio(self):
        # Adicione a lógica para processar ou reproduzir o áudio gravado
        if self.recorded_audio:
            file_path = "recorded_audio.wav"  # Nome do arquivo desejado
            try:
                self.audio_data, self.sampling_rate = librosa.load(file_path, sr=None)
                print(f"Áudio carregado com sucesso. Taxa de amostragem: {self.sampling_rate}")
            except Exception as e:
                print(f"Erro ao carregar o áudio: {e}")
        else:
            tk.messagebox.showinfo("Aviso", "Grave arquivo de áudio primeiro.")

   
    def record_audio(self):
        # Função para gravar áudio com taxa de amostragem configurável e tempo de aquisição ajustável
        sampling_rate = int(self.sampling_rate_var.get())
        duration = self.duration_var.get()
        recording = sd.rec(int(sampling_rate * duration), samplerate=sampling_rate, channels=1, dtype=np.int16)
        sd.wait()

        # Salvar o áudio gravado em um arquivo WAV
        write('recorded_audio.wav', sampling_rate, recording)
        self.recorded_audio = True
        print("Áudio gravado com sucesso!")


    def load_audio(self):
        file_path = filedialog.askopenfilename(title="Selecione um arquivo de áudio", filetypes=[("Arquivos de Áudio", "*.wav;*.mp3")])
        if file_path:
            self.audio_data, self.sampling_rate = librosa.load(file_path, sr=None)
    
    def apply_rectangular_window(self,signal, window_size):
        return signal[:window_size]

    def apply_hamming_window(self,signal, window_size):
        hamming_window = np.hamming(window_size)
        return signal[:window_size] * hamming_window

    def apply_hanning_window(self,signal, window_size):
        hanning_window = np.hanning(window_size)
        return signal[:window_size] * hanning_window

    def apply_triangular_window(self,signal, window_size):
        triangular_window = np.bartlett(window_size)
        return signal[:window_size] * triangular_window

    def apply_flat_top_window(self,signal, window_size):
        flat_top_window = flattop(window_size)
        return signal[:window_size] * flat_top_window
    
    def plot_mfccs(self):
        def plot_mfccs(signal, sampling_rate):
            mfccs = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13)
            
            # Plotar os coeficientes cepstrais de frequência mel (MFCCs)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
            plt.title('MFCCs')
            plt.show()
        # Adicione a lógica para plotar os coeficientes cepstrais de frequência mel (MFCCs)
        if self.audio_data is not None:

            try:
                windowed_signal = self.apply_window(self.audio_data, int(self.window_size_var.get()))
                # Chamar a função plot_mfccs com o sinal e taxa de amostragem
                plot_mfccs(windowed_signal, self.sampling_rate)
            except Exception as e:
                print(f"Erro ao plotar MFCCs: {e}")
        else:
            tk.messagebox.showinfo("Aviso", "Carregue um arquivo de áudio primeiro.")
    
    def plot_signal_and_spectrum(self,signal, sampling_rate):
        # Verificar se o comprimento do sinal é válido
        if len(signal) == 0:
            print("Erro: Comprimento do sinal é zero.")
            return

    
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

    def apply_window(self, signal, window_size):
        if(window_size>len(signal)):
            window_size = len(signal)
            
        if self.selected_window_type.get() == "hamming":
            return self.apply_hamming_window(signal, window_size)
        elif self.selected_window_type.get() == "hanning":
            return self.apply_hanning_window(signal, window_size)
        elif self.selected_window_type.get() == "rectangular":
            return self.apply_rectangular_window(signal, window_size)
        elif self.selected_window_type.get() == "triangular":
            return self.apply_triangular_window(signal, window_size)
        elif self.selected_window_type.get() == "flat_top":
            return self.apply_flat_top_window(signal, window_size)
        else:
            return signal  # Caso nenhum tipo de janela seja selecionado, retornar o sinal original

    def plot_graph(self):
        if hasattr(self, "audio_data"):
            windowed_signal = self.apply_window(self.audio_data, int(self.window_size_var.get()))

            # Verificar se o comprimento do sinal é válido antes de calcular a FFT
            self.plot_signal_and_spectrum(windowed_signal, self.sampling_rate)
        else:
            tk.messagebox.showinfo("Aviso", "Carregue um arquivo de áudio primeiro.")

# Inicializar a aplicação
root = tk.Tk()
app = AudioAnalyzerApp(root)
root.mainloop()
