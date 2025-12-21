import tkinter as tk

class MainUserScreen(tk.Frame):
    def __init__(self, userScreen, controller):
        super().__init__(userScreen, bg ="white")
        self.controller = controller
        self.pack(fill="both", expand=True)
        tk.Label(self,
                 text="Digits Classification Problems",
                 font=("Roboto", 30, "bold"),
                 bg="white").pack(pady=150)
        
        tk.Button(self,
                  text="Draw and Guess",
                  font=("Roboto", 15, "bold"),
                  width=30, height=3,borderwidth=4, relief="solid",
                  bg="white", fg="black",
                  command=self.controller.DrawAndGuessMenu).pack(pady=30)

        tk.Button(self,
                  text="Run random MNIST sample",
                  font=("Roboto", 15, "bold"),
                  width=30, height=3, borderwidth=4, relief="solid",
                  bg="white", fg="black",
                  command=self.controller.DrawTestMNIST).pack(pady=30)

        tk.Button(self,
                  text="Exit",
                  font=("Roboto", 15, "bold"),
                  width=30, height=3, borderwidth=4, relief="solid",
                  bg="white", fg="black",
                  command=userScreen.destroy).pack(pady=30)
