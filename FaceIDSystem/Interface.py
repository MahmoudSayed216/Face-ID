import cv2
import tkinter as tk
import customtkinter as ctk
import threading
from PIL import Image, ImageTk
from FaceIDSystem.widgtes.ModuleContainer import ModuleContainer
from FaceIDSystem.widgtes.ScrollableFrame import ScrollableFrame

class CameraViewer:
    def __init__(self, root: tk.Tk, function=None, src=0):
        self.root = root
        self.root.title("Face ID")
        self.root.geometry("1280x720")
        self.present_identities = {}
        self._initialize_interface()
        self.cap = cv2.VideoCapture(src)
        self.function = function or (lambda f: f)

        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.current_faces = {}

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    def _initialize_interface(self):
        self.camera_area = tk.Label(self.root)
        self.camera_area.pack()

        self.scrollable_frame = ScrollableFrame(self.root, 300, 600)
        


        

    def _on_mouse_wheel(self, event):
        direction = -1 if event.num == 4 else 1
        self.scrollable_frame._parent_canvas.yview_scroll(direction, "units")
        
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest_frame = frame
    def _update_scrollable_frame(self):
        self.scrollable_frame.update_content(self.current_faces)


    def _tick(self):
        with self.lock:
            frame = self.latest_frame

        if frame is not None:
            frame = cv2.imread("./input/MandA.jpeg")
            annotated, current_faces = self.function(frame)
            if set(self.current_faces.keys()).symmetric_difference(set(current_faces.keys())): #FIXME: CHECK IF AN EMPTY LIST IS INTERPRETED AS FALSE AND VV
                self.current_faces = current_faces
                self._update_scrollable_frame()

            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(annotated))
            self.camera_area.imgtk = img
            self.camera_area.configure(image=img)

        if self.running:
            self.root.after(15, self._tick)

    def close(self):
        self.running = False
        self.capture_thread.join(timeout=1)
        self.cap.release()
        self.root.destroy()