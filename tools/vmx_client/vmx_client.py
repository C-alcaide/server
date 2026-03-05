import tkinter as tk
from tkinter import messagebox, ttk
import socket
import threading
import time

class CasparClient:
    def __init__(self, master):
        self.master = master
        master.title("CasparCG VMX Client")
        master.geometry("400x600")

        self.sock = None
        self.connected = False
        self.last_speed_time = 0

        # Styles
        style = ttk.Style()
        style.configure("TButton", padding=6)
        style.configure("TEntry", padding=6)
        style.configure("TLabel", padding=6)

        # Connection Frame
        self.conn_frame = ttk.LabelFrame(master, text="Connection")
        self.conn_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self.conn_frame, text="Host:").pack(side="left")
        self.host_entry = ttk.Entry(self.conn_frame, width=15)
        self.host_entry.insert(0, "127.0.0.1")
        self.host_entry.pack(side="left", padx=5)

        ttk.Label(self.conn_frame, text="Port:").pack(side="left")
        self.port_entry = ttk.Entry(self.conn_frame, width=6)
        self.port_entry.insert(0, "5250")
        self.port_entry.pack(side="left", padx=5)

        self.connect_btn = ttk.Button(self.conn_frame, text="Connect", command=self.toggle_connect)
        self.connect_btn.pack(side="left", padx=5)

        # Channel/Layer Frame
        self.target_frame = ttk.LabelFrame(master, text="Designation")
        self.target_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(self.target_frame, text="Channel:").pack(side="left")
        self.chan_entry = ttk.Entry(self.target_frame, width=5)
        self.chan_entry.insert(0, "1")
        self.chan_entry.pack(side="left", padx=5)

        ttk.Label(self.target_frame, text="Layer:").pack(side="left")
        self.layer_entry = ttk.Entry(self.target_frame, width=5)
        self.layer_entry.insert(0, "10")
        self.layer_entry.pack(side="left", padx=5)

        # File Frame
        self.file_frame = ttk.LabelFrame(master, text="Media")
        self.file_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(self.file_frame, text="File:").pack(side="left")
        self.file_entry = ttk.Entry(self.file_frame, width=25)
        self.file_entry.insert(0, "my_recording")
        self.file_entry.pack(side="left", padx=5, fill="x", expand=True)

        self.play_btn = ttk.Button(self.file_frame, text="PLAY", command=self.send_play)
        self.play_btn.pack(side="right", padx=5)

        # Controls Frame
        self.ctrl_frame = ttk.LabelFrame(master, text="Controls")
        self.ctrl_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(self.ctrl_frame, text="STOP", command=self.send_stop).pack(side="left", padx=5)
        ttk.Button(self.ctrl_frame, text="PAUSE (Speed 0)", command=lambda: self.set_speed(0)).pack(side="left", padx=5)
        ttk.Button(self.ctrl_frame, text="SEEK LIVE", command=self.send_seek_live).pack(side="left", padx=5)

        # Speed Frame (Using scale for range -30 to 30)
        self.speed_frame = ttk.LabelFrame(master, text="Speed (-30.0 to 30.0)")
        self.speed_frame.pack(fill="x", padx=10, pady=5)

        self.speed_var = tk.DoubleVar(value=1.0)
        
        # Display current value
        self.speed_label = ttk.Label(self.speed_frame, text="Speed: 1.00")
        self.speed_label.pack()

        # Scale slider
        self.speed_scale = ttk.Scale(self.speed_frame, from_=-30.0, to=30.0, variable=self.speed_var, command=self.on_speed_change)
        self.speed_scale.pack(fill="x", padx=10, pady=5)

        # Reset button
        ttk.Button(self.speed_frame, text="Reset Speed (1.0)", command=lambda: self.set_speed(1.0)).pack(pady=5)

        # Slow Motion Frame
        self.slow_frame = ttk.LabelFrame(master, text="Slow Motion")
        self.slow_frame.pack(fill="x", padx=10, pady=5)

        ttk.Button(self.slow_frame, text="-0.5x", command=lambda: self.set_speed(-0.5)).pack(side="left", padx=5)
        ttk.Button(self.slow_frame, text="-0.25x", command=lambda: self.set_speed(-0.25)).pack(side="left", padx=5)
        ttk.Button(self.slow_frame, text="-0.1x", command=lambda: self.set_speed(-0.1)).pack(side="left", padx=5)
        
        ttk.Button(self.slow_frame, text="0.1x", command=lambda: self.set_speed(0.1)).pack(side="left", padx=5)
        ttk.Button(self.slow_frame, text="0.25x", command=lambda: self.set_speed(0.25)).pack(side="left", padx=5)
        ttk.Button(self.slow_frame, text="0.5x", command=lambda: self.set_speed(0.5)).pack(side="left", padx=5)

        # Log
        self.log_text = tk.Text(master, height=10, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

    def log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def toggle_connect(self):
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        host = self.host_entry.get()
        try:
            port = int(self.port_entry.get())
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set timeout only for connection phase
            self.sock.settimeout(3.0)
            self.sock.connect((host, port))
            # Remove timeout for persistent connection so recv doesn't throw
            self.sock.settimeout(None)
            
            self.connected = True
            self.connect_btn.config(text="Disconnect")
            self.log(f"Connected to {host}:{port}")
            
            # Start reader thread
            threading.Thread(target=self.socket_reader, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Connection Error", str(e))
            self.log(f"Error: {e}")

    def disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None
        self.connected = False
        self.connect_btn.config(text="Connect")
        self.log("Disconnected")

    def socket_reader(self):
        while self.connected and self.sock:
            try:
                data = self.sock.recv(4096)
                if not data:
                    break
                # Simple log of received data
                # self.log("RX: " + data.decode("utf-8").strip())
            except:
                break
        if self.connected:
            self.master.after(0, self.disconnect)

    def send_cmd(self, cmd):
        if not self.connected:
            self.log("Not connected!")
            return
        try:
            cmd_bytes = (cmd + "\r\n").encode("utf-8")
            self.sock.sendall(cmd_bytes)
            self.log(f"TX: {cmd}")
        except Exception as e:
            self.log(f"Send Error: {e}")
            self.disconnect()

    def get_chan_layer(self):
        return f"{self.chan_entry.get()}-{self.layer_entry.get()}"

    def send_play(self):
        target = self.get_chan_layer()
        # VMX PLAY command structure
        cmd = f"PLAY {target} {self.file_entry.get()}"
        self.send_cmd(cmd)
        # Reset speed on play? Usually good practice
        self.set_speed(1.0)

    def send_stop(self):
        target = self.get_chan_layer()
        self.send_cmd(f"STOP {target}")

    def send_seek_live(self):
        target = self.get_chan_layer()
        self.send_cmd(f"CALL {target} SEEK LIVE")
        # Ensure speed is playing forward
        self.set_speed(1.0)

    def set_speed(self, value):
        self.speed_var.set(value)
        self.on_speed_change(value, force=True)

    def on_speed_change(self, val, force=False):
        # Throttle updates to avoid flooding
        now = time.time()
        if not force and (now - self.last_speed_time < 0.1):
            return
        
        self.last_speed_time = now
        speed_val = float(val)
        self.speed_label.config(text=f"Speed: {speed_val:.2f}")
        
        target = self.get_chan_layer()
        # VMX SPEED command
        self.send_cmd(f"CALL {target} SPEED {speed_val:.3f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CasparClient(root)
    root.mainloop()
