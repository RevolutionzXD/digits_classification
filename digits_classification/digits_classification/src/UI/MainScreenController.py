import tkinter as tk
from .MainUserScreen import MainUserScreen
from .DrawAndGuess import DrawAndGuessScreen
from .TestMNISTUI import TestMNISTScreen


class UserController:

    def ExitScreen(self, event = None):
        self.userScreen.destroy()
        
    def __init__(self):
        self.userScreen = tk.Tk()
        self.userScreen.attributes("-fullscreen", True)
        self.userScreen.config(bg="White")

        self.currentScreen = None
        self.DrawMainUserScreen()

        self.userScreen.bind("<Escape>", self.ExitScreen)
        self.userScreen.mainloop()


    def DrawMainUserScreen(self):
        self.ClearScreen()
        self.currentScreen = MainUserScreen(self.userScreen, self)
        

    def DrawAndGuessMenu(self):
        self.ClearScreen()
        self.currentScreen = DrawAndGuessScreen(self.userScreen, self)

    def ClearScreen(self):
        if self.currentScreen != None:
            self.currentScreen.destroy()
    def DrawTestMNIST(self):
        self.ClearScreen()
        self.currentScreen = TestMNISTScreen(self.userScreen, self)
