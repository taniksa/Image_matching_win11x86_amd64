import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class ImageMatcherApp:
    def __init__(self, master):
        self.master = master
        master.title("Image Matcher")

        # Initialize variables
        self.input_image = None
        self.directory_path = None
        self.matches = []
        self.current_match_index = -1

        # Create input image selection widgets
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(pady=10)
        self.input_label = tk.Label(self.input_frame, text="Select Input Image:")
        self.input_label.grid(row=0, column=0)
        self.input_entry = tk.Entry(self.input_frame, width=40)
        self.input_entry.grid(row=0, column=1)
        self.input_button = tk.Button(self.input_frame, text="Browse", command=self.browse_image)
        self.input_button.grid(row=0, column=2)

        # Create directory selection widgets
        self.directory_frame = tk.Frame(master)
        self.directory_frame.pack(pady=10)
        self.directory_label = tk.Label(self.directory_frame, text="Select Directory:")
        self.directory_label.grid(row=0, column=0)
        self.directory_entry = tk.Entry(self.directory_frame, width=40)
        self.directory_entry.grid(row=0, column=1)
        self.directory_button = tk.Button(self.directory_frame, text="Browse", command=self.browse_directory)
        self.directory_button.grid(row=0, column=2)

        # Create button to find match
        self.match_button = tk.Button(master, text="Find Match", command=self.find_match)
        self.match_button.pack(pady=10)

        # Create label to display result
        self.result_label = tk.Label(master, text="")
        self.result_label.pack(pady=10)

        # Create image display frame
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(pady=10)

        # Create next and previous buttons
        self.prev_button = tk.Button(master, text="Previous", state="disabled", command=self.show_previous)
        self.prev_button.pack(side="left", padx=10)
        self.next_button = tk.Button(master, text="Next", state="disabled", command=self.show_next)
        self.next_button.pack(side="right", padx=10)

    def browse_image(self):
        filename = filedialog.askopenfilename()
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, filename)
        self.input_image = filename

    def browse_directory(self):
        directory_path = filedialog.askdirectory()
        self.directory_entry.delete(0, tk.END)
        self.directory_entry.insert(0, directory_path)
        self.directory_path = directory_path

    def find_match(self):
        self.matches = []
        if self.input_image and self.directory_path:
            # Load input image
            input_img = cv2.imread(self.input_image, 0)

            # Initiate ORB detector
            orb = cv2.ORB_create()

            # Find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(input_img, None)

            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Iterate through files in directory
            for filename in os.listdir(self.directory_path):
                filepath = os.path.join(self.directory_path, filename)
                # Load image from directory
                directory_img = cv2.imread(filepath, 0)

                # Find the keypoints and descriptors with ORB
                kp2, des2 = orb.detectAndCompute(directory_img, None)

                # Match descriptors
                matches = bf.match(des1, des2)

                # Calculate distance
                distance = sum(match.distance for match in matches) / len(matches)

                # Store match details
                self.matches.append((filename, distance))

            # Sort matches by distance
            self.matches.sort(key=lambda x: x[1])

            # Display first match
            self.current_match_index = 0
            self.display_match()

    def display_match(self):
        if self.matches:
            match_filename, _ = self.matches[self.current_match_index]
            full_path = os.path.join(self.directory_path, match_filename)
            self.result_label.config(text="Exact match found: " + full_path)

            # Load input image and matched image
            input_img = cv2.imread(self.input_image)
            matched_img = cv2.imread(full_path)

            # Convert images to RGB format
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

            # Convert images to PIL format
            input_img_pil = Image.fromarray(input_img)
            matched_img_pil = Image.fromarray(matched_img)

            # Resize images to fit the window without cropping
            max_width = min(self.master.winfo_screenwidth() // 2 - 50, input_img_pil.width, matched_img_pil.width)
            max_height = min(self.master.winfo_screenheight() - 200, input_img_pil.height, matched_img_pil.height)
            input_img_pil = input_img_pil.resize((max_width, max_height))
            matched_img_pil = matched_img_pil.resize((max_width, max_height))

            # Convert images to tkinter format
            input_img_tk = ImageTk.PhotoImage(input_img_pil)
            matched_img_tk = ImageTk.PhotoImage(matched_img_pil)

            # Update image display
            if hasattr(self, "input_img_label"):
                self.input_img_label.config(image=input_img_tk)
                self.input_img_label.image = input_img_tk
            else:
                self.input_img_label = tk.Label(self.image_frame, image=input_img_tk)
                self.input_img_label.image = input_img_tk
                self.input_img_label.pack(side="left")

            if hasattr(self, "matched_img_label"):
                self.matched_img_label.config(image=matched_img_tk)
                self.matched_img_label.image = matched_img_tk
            else:
                self.matched_img_label = tk.Label(self.image_frame, image=matched_img_tk)
                self.matched_img_label.image = matched_img_tk
                self.matched_img_label.pack(side="right")

            # Enable next and previous buttons if there are more than one matches
            if len(self.matches) > 1:
                self.prev_button.config(state="normal")
                self.next_button.config(state="normal")
            else:
                self.prev_button.config(state="disabled")
                self.next_button.config(state="disabled")

    def show_previous(self):
        if self.current_match_index > 0:
            self.current_match_index -= 1
            self.display_match()

    def show_next(self):
        if self.current_match_index < len(self.matches) - 1:
            self.current_match_index += 1
            self.display_match()

def main():
    root = tk.Tk()
    app = ImageMatcherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
