import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sounddevice as sd
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
import os
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading

# ---------------------- UI COLORS ----------------------
c1 = "#e63946"
c2 = "#f1faee"
c3 = "#a8dadc"
c4 = "#1d3557"
w, h = 700, 1000  # Wider layout

# ---------------------- GLOBALS ----------------------
fs = 8000
live_buffer = np.zeros(fs * 5)
stream = None  # Prevent garbage collection

# ---------------------- APP SETUP ----------------------
root = tk.Tk()
root.geometry(f'{w}x{h}')
root.title("CMC EKOBIT")
root.config(bg=c2)
root.columnconfigure(0, weight=1)
root.iconbitmap("heart.ico")
root.maxsize(height=h,width=w)
root.minsize(height=h,width=w)

# ---------------------- VARIABLES ----------------------
name_var = tk.StringVar()
organ_var = tk.StringVar(value="Heart")
device_var = tk.StringVar()
duration_var = tk.StringVar(value="10")
lpf_var = tk.StringVar(value="200")
folder_path = tk.StringVar(value=os.getcwd())

# ---------------------- FILTERS ----------------------
def notch_filter(data, freq=50, fs=8000, Q=30):
    b, a = signal.iirnotch(freq, Q, fs)
    return signal.filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs=8000, order=6):
    b, a = signal.butter(order, cutoff / (0.5 * fs), btype='low')
    return signal.filtfilt(b, a, data)

# ---------------------- RECORDING FUNCTION ----------------------
def record_audio():
    name = name_var.get().strip()
    organ = organ_var.get()
    duration = int(duration_var.get())
    cutoff = int(lpf_var.get())
    folder = folder_path.get()

    if not name:
        messagebox.showerror("Missing Input", "Please enter a name.")
        return
    if not device_var.get():
        messagebox.showerror("Missing Input", "Please select an input device.")
        return

    def thread_record():
        try:
            index = devices.index(device_var.get())
            messagebox.showinfo("Recording", f"Recording {duration} sec at 8kHz...")
            audio = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='float32', device=index)
            sd.wait()
            audio = audio[:, 0]
            audio_notched = notch_filter(audio, fs=fs)
            audio_filtered = lowpass_filter(audio_notched, cutoff=cutoff, fs=fs)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{name}_{organ}_{timestamp}"
            wav.write(os.path.join(folder, f"{base}_raw.wav"), fs, audio.astype(np.float32))
            wav.write(os.path.join(folder, f"{base}_filtered.wav"), fs, audio_filtered.astype(np.float32))
            messagebox.showinfo("Saved", f"Saved as:\n{base}_raw.wav\n{base}_filtered.wav")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    threading.Thread(target=thread_record, daemon=True).start()

# ---------------------- UI LAYOUT ----------------------

tk.Label(root, text="CMC EKOBIT", font=("Oswald Medium", 28), fg=c2, bg=c4).pack(fill="x", pady=(0, 10))

# ---------------------- Name ----------------------
name_frame = tk.LabelFrame(root, text="Patient Name", bg=c2, font=("Oswald", 10))
name_frame.pack(fill="x", padx=10, pady=5)
tk.Entry(name_frame, textvariable=name_var, font=("Oswald", 12)).pack(padx=5, pady=5, fill="x")

# ---------------------- Organ & Input ----------------------
option_frame = tk.LabelFrame(root, text="Organ & Audio Input", bg=c2, font=("Oswald", 10))
option_frame.pack(fill="x", padx=10, pady=5)

tk.Label(option_frame, text="Organ", bg=c2, font=("Oswald", 10)).grid(row=0, column=0, padx=5, sticky='w')
ttk.Combobox(option_frame, textvariable=organ_var, values=["Heart", "Lungs"], state="readonly", width=30).grid(row=0, column=1, padx=5)

tk.Label(option_frame, text="Input Device", bg=c2, font=("Oswald", 10)).grid(row=1, column=0, padx=5, sticky='w')
devices = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]
if not devices:
    messagebox.showerror("No input devices", "No audio input devices found!")
    root.destroy()
default_input = sd.default.device[0] if sd.default.device else 0
device_var.set(devices[default_input] if default_input < len(devices) else devices[0])
ttk.Combobox(option_frame, textvariable=device_var, values=devices, state="readonly", width=30).grid(row=1, column=1, padx=5)

# ---------------------- Settings ----------------------
settings_frame = tk.LabelFrame(root, text="Recording Settings", bg=c2, font=("Oswald", 10))
settings_frame.pack(fill="x", padx=10, pady=5)

tk.Label(settings_frame, text="Duration (s)", bg=c2, font=("Oswald", 10)).grid(row=0, column=0, padx=5, sticky='w')
ttk.Combobox(settings_frame, textvariable=duration_var, values=["10", "15", "20", "30", "60"], state="readonly", width=30).grid(row=0, column=1, padx=5)

tk.Label(settings_frame, text="LPF Cutoff (Hz)", bg=c2, font=("Oswald", 10)).grid(row=1, column=0, padx=5, sticky='w')
ttk.Combobox(settings_frame, textvariable=lpf_var, values=["200", "500", "1000"], state="readonly", width=30).grid(row=1, column=1, padx=5)

# ---------------------- Folder ----------------------
folder_frame = tk.LabelFrame(root, text="Save Location", bg=c2, font=("Oswald", 10))
folder_frame.pack(fill="x", padx=10, pady=5)
tk.Button(folder_frame, text="Browse Folder", command=lambda: folder_path.set(filedialog.askdirectory()), font=("Oswald", 10), bg=c3).pack(padx=5, pady=3)
tk.Label(folder_frame, textvariable=folder_path, bg=c2, font=("Oswald", 8), wraplength=w-20).pack(padx=5)

# ---------------------- Record Button ----------------------
tk.Button(root, text="Start Recording", command=record_audio, font=("Oswald", 12), bg=c1, fg="white").pack(fill="x", padx=20, pady=10)

# ---------------------- Graph Area (Bottom) ----------------------
graph_frame = tk.LabelFrame(root, text="Live Waveform (Buffered)", bg=c2, font=("Oswald", 10))
graph_frame.pack(fill="both", expand=True, padx=10, pady=5)

fig, ax = plt.subplots(figsize=(7, 2), dpi=100)
plot_line, = ax.plot(live_buffer, color='blue', linewidth=0.8)
ax.set_title("Stethescope Signal", fontsize=10)
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")
ax.grid(True)
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack(fill="both", expand=True)

# ---------------------- Footer ----------------------
tk.Label(root, text="Designed by Shreenandan Sahu 💓", bg=c2, font=("Oswald", 9)).pack(pady=3)

# ---------------------- Live Stream ----------------------
def start_live_waveform():
    global stream
    def audio_callback(indata, frames, time, status):
        global live_buffer
        try:
            data = indata[:, 0]
            live_buffer[:] = np.roll(live_buffer, -frames)
            live_buffer[-frames:] = data
            plot_line.set_ydata(live_buffer)
            canvas.draw_idle()
        except:
            pass  # don't crash on update

    try:
        index = devices.index(device_var.get())
        stream = sd.InputStream(samplerate=fs, channels=1, callback=audio_callback, device=index)
        stream.start()
    except Exception as e:
        print("Live stream error:", e)

root.after(1000, start_live_waveform)
root.mainloop()


