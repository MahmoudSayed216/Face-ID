from FaceIDSystem.Interface import CameraViewer
import tkinter as tk
from typing import Callable


class CameraViewerWrapper:
    def __init__(self, function: Callable):
        self.function = function
        # self.interface = Interface
    def run(self):
        root = tk.Tk()
        interface = CameraViewer(root, function= self.function)
        root.mainloop()