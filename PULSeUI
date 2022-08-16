import tkinter
import numpy as np
import customtkinter
import os
import pendulum
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATH = os.path.dirname(os.path.realpath(__file__))


class StartFrame(customtkinter.CTk):

    # Parameters
    APP_NAME = "PULSe Screensaver"
    WIDTH = 640
    HEIGHT = 380

    # Initialize
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(StartFrame.APP_NAME)
        self.geometry(f"{StartFrame.WIDTH}x{StartFrame.HEIGHT}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.minsize(StartFrame.WIDTH, StartFrame.HEIGHT)
        self.maxsize(StartFrame.WIDTH, StartFrame.HEIGHT)
        self.resizable(False, False)

        self.background_image = tkinter.PhotoImage(file='images/screensaver.png')
        self.background_label = tkinter.Label(self, image=self.background_image)
        self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.button = customtkinter.CTkButton(master=self,
                                              text="START",
                                              corner_radius=25,  # <- custom border_width
                                              bg_color='#00407a',
                                              command=lambda: self.button_event(),
                                              text_color='#FFFFFF',
                                              text_font=("Roboto Medium", -13))
        self.button.pack(pady=180, padx=10)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Methods
    def on_closing(self, event=0):
        self.destroy()
        root.destroy()

    def start(self):
        self.mainloop()

    def button_event(self):
        top.withdraw()
        root.deiconify()


class MainFrame(customtkinter.CTk):

    APP_NAME = "PULSe UI"
    WIDTH = 640
    HEIGHT = 380

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(MainFrame.APP_NAME)
        self.geometry(f"{MainFrame.WIDTH}x{MainFrame.HEIGHT}")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.minsize(MainFrame.WIDTH, MainFrame.HEIGHT)
        self.maxsize(MainFrame.WIDTH, MainFrame.HEIGHT)
        self.resizable(False, False)

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=MainFrame.WIDTH,
                                            height=MainFrame.HEIGHT,
                                            corner_radius=15,
                                            fg_color='#f5f8fe')

        self.frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.frame.grid_rowconfigure(2, weight=1)
        self.frame.grid_columnconfigure(2, weight=1)

        # Frame layout - see image
        self.frame_menu = self.make_frame(master=self.frame, width=150, height=360, corner_radius=15,
                                          row=0, column=0, columnspan=1, rowspan=3, padx=10, pady=10,
                                          sticky='nsew', fg_color="#00407a", border_color='#e6e6e6', border_width=2)

        self.frame_header = self.make_frame(master=self.frame, width=400, height=10, corner_radius=15,
                                            row=0, column=1, columnspan=3, rowspan=1, padx=10, pady=10,
                                            sticky='w')

        self.frame_info = self.make_frame(master=self.frame, width=250, height=90, corner_radius=15,
                                          row=1, column=1, columnspan=2, rowspan=1, padx=10, pady=5,
                                          sticky='w')

        self.frame_graph = self.make_frame(master=self.frame, width=400, height=310, corner_radius=15,
                                           row=2, column=1, columnspan=3, rowspan=1, padx=10, pady=10,
                                           sticky='w', fg_color="#FFFFFF", border_color='#e6e6e6', border_width=2)

        # Menu Frame: Hello User, Start measurement button

        self.frame_menu.grid_rowconfigure(3,weight=1)
        self.frame_menu.grid_columnconfigure(0, weight=1)

        self.welcome = customtkinter.CTkLabel(master=self.frame_menu,
                                              width=125, height=25,
                                              corner_radius=15,
                                              text='Hello user,',
                                              text_font=("Roboto Medium", -22),
                                              text_color='#FFFFFF',
                                              anchor='w')

        self.welcome.grid(row=0, column=0, sticky='ne', padx=10, pady=(10,0))

        self.date = customtkinter.CTkLabel(master=self.frame_menu,
                                           width=125, height=15,
                                           corner_radius=15,
                                           text=pendulum.now().to_formatted_date_string(),
                                           text_font=("Roboto Medium", -13),
                                           text_color='#FFFFFF',
                                           anchor='w')

        self.date.grid(row=1, column=0, sticky='n', padx=10, pady=(0, 10))

        self.start_intensity = customtkinter.CTkButton(master=self.frame_menu,
                                                       width=125, height=25,
                                                       corner_radius=15,
                                                       text='Intensity',
                                                       text_font=("Roboto Medium", -13),
                                                       fg_color='#FFFFFF',
                                                       commmand = self.intensity())

        self.start_intensity.grid(row=2, column=0, sticky='n', padx=10, pady=(15, 0))

        self.start_meas = customtkinter.CTkButton(master=self.frame_menu,
                                                  width=125, height=25,
                                                  corner_radius=15,
                                                  text='Measure',
                                                  text_font=("Roboto Medium", -13),
                                                  fg_color='#FFFFFF',
                                                  command=self.measure())

        self.start_meas.grid(row=3, column=0, sticky='n', padx=10, pady=(15, 75))

        # Header Frame: Blank  - Blank - Logo:

        self.frame_header.grid_rowconfigure(0, weight=1)
        self.frame_header.grid_columnconfigure(2, weight=1)

        self.pulse_logo = customtkinter.CTkButton(master=self.frame_header,
                                                  width=25, height=25,
                                                  text="Logout",
                                                  corner_radius=10,
                                                  text_font=("Roboto Medium", 11),
                                                  text_color='#FFFFFF',
                                                  command=lambda: self.logo_return())
        self.pulse_logo.grid(row=0, column=2, padx=(325, 10), pady=0)
        # Info Frame: Class  - Concentration:

        self.frame_info.grid_rowconfigure(0, weight=1)
        self.frame_info.grid_columnconfigure(1, weight=1)

        self.sepsis_class = customtkinter.CTkLabel(master=self.frame_info,
                                                   width=180, height=80,
                                                   text='Class:',
                                                   corner_radius=15,
                                                   text_font=("Roboto Medium", 11),
                                                   fg_color='#00407a',
                                                   text_color='#FFFFFF',
                                                   anchor=tkinter.NW)
        self.sepsis_class.grid(row=0, column=0, padx=0, pady=5)

        self.sepsis_class_value = customtkinter.CTkLabel(master=self.sepsis_class,
                                                         width=50, height=50,
                                                         text='Systemic Inflammation Stage 2',
                                                         corner_radius=15,
                                                         text_font=("Roboto Medium", 10),
                                                         text_color='#FFFFFF',
                                                         anchor=tkinter.W,
                                                         wraplength=180)
        self.sepsis_class_value.grid(row=0, column=0, padx=0, pady=(10, 10))

        self.sepsis_conc = customtkinter.CTkLabel(master=self.frame_info,
                                                  width=180, height=80,
                                                  text='Concentration:',
                                                  corner_radius=15,
                                                  text_font=("Roboto Medium", 11),
                                                  fg_color='#00407a',
                                                  text_color='#FFFFFF',
                                                  anchor=tkinter.NW)
        self.sepsis_conc.grid(row=0, column=1, padx=(10, 75), pady=5)

        self.sepsis_conc_value = customtkinter.CTkLabel(master=self.sepsis_conc,
                                                        width=50, height=50,
                                                        text='133 pg/mL',
                                                        corner_radius=15,
                                                        text_font=("Roboto Medium", 10),
                                                        text_color='#FFFFFF',
                                                        anchor=tkinter.W,
                                                        wraplength=180)
        self.sepsis_conc_value.grid(row=0, column=0, padx=0, pady=(10, 10))

        # Graph Frame with matplotlib:
        figure = Figure(figsize=(4, 3), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self.frame_graph)
        axes = figure.add_subplot()
        axes.plot([i for i in range(50)], [np.sin(i) for i in range(50)], c='#00407a')
        figure_canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Protocols:
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self, event=0):
        self.destroy()
        top.destroy()

    def start(self):
        self.mainloop()

    def make_frame(self, master, width, height, corner_radius, row, column, columnspan, rowspan, padx, pady,
                   sticky, border_width=0, border_color=None, fg_color=None):

        frame = customtkinter.CTkFrame(master=master,
                                       width=width,
                                       height=height,
                                       corner_radius=corner_radius,
                                       border_color=border_color,
                                       border_width=border_width,
                                       fg_color=fg_color)

        frame.grid(row=row, column=column,
                   columnspan=columnspan,
                   rowspan=rowspan,
                   padx=padx, pady=pady,
                   sticky=sticky)

        return frame

    def intensity(self):
        return None

    def measure(self):
        return None

    def logo_return(self):
        root.withdraw()
        top.deiconify()

    def load_image(self, path, image_size):
        return ImageTk.PhotoImage(Image.open(PATH + path).resize((image_size, image_size)))


if __name__ == '__main__':
    top = StartFrame()
    root = MainFrame()

    show_start = True   # Turn off when working on the main screen

    if show_start:
        top.start()
    else:
        root.start()
