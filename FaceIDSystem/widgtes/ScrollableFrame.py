import tkinter as tk
import customtkinter
# from widgets.ModuleContainer import ModuleContainer
# from resources.colors import *
from FaceIDSystem.widgtes.ModuleContainer import ModuleContainer

class ScrollableFrame:

	def __init__(self, master, width, height):
		# self.items = items

		self.scrollable_frame = customtkinter.CTkScrollableFrame(master, corner_radius=0, width=width, height=height)
		self.scrollable_frame.place(relx=0.8, rely=0, anchor='n')
		self.scrollable_frame.bind("<Button-4>", self._on_mouse_wheel)
		self.scrollable_frame.bind("<Button-5>", self._on_mouse_wheel)
		self.module_containers_refs = []
		self.pack_infos = []
		# self.list_items()

	def update_content(self, current_faces):
		for k, v in current_faces.items():
			# print(item.module_name)
			# ref1 = ModuleContainer(self.scrollable_frame, bg=WHITE, width=180, height=180, module_data=item)
			ref1 = ModuleContainer(self.scrollable_frame, bg="#ffffff", width=150, height=200, image=v["face"], name=v["name"], _id=k)
			self.module_containers_refs.append(ref1)
			self.pack_infos.append(ref1.frame.pack_info())
					
			# ref2 = ModuleContainer(self.scrollable_frame, bg=WHITE, width=180, height=180, module_data=item)
			# self.module_containers_refs.append(ref2)
			# self.pack_infos.append(ref2.frame.pack_info())

			# ref3 = ModuleContainer(self.scrollable_frame, bg=WHITE, width=180, height=180, module_data=item)
			# self.module_containers_refs.append(ref3)
			# self.pack_infos.append(ref3.frame.pack_info())


	def _on_mouse_wheel(self, event):
			direction = -1 if event.num == 4 else 1
			self.scrollable_frame._parent_canvas.yview_scroll(direction, "units")

	def hide_unhide_containers(self, search_text):
		
		if search_text == "":
			for ref, pack_info in zip(self.module_containers_refs, self.pack_infos):
				ref.show(pack_info)
			return
		
		for ref, pack_info in zip(self.module_containers_refs, self.pack_infos):
			if search_text.lower() in ref.module_data.module_name.lower():
				ref.show(pack_info)
			else:
				ref.hide()
