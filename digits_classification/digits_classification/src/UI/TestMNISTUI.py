import tkinter as tk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import random

from src.infer import loadModel


class TestMNISTScreen(tk.Frame):
    def __init__(self, userScreen, controller):
        super().__init__(userScreen, bg="white")
        self.controller = controller
        self.pack(fill="both", expand=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST("data", train=False, download=True, transform=self.transform)
        self.curIdx = 0

        #Models
        self.modelMLP = None
        self.modelCNN = None
        self.activeModel = None
        self.activeModelName = tk.StringVar(value="Model: (chưa chọn)")

        #Title
        tk.Label(self, text="Test MNIST", font=("Roboto", 28, "bold"), bg="white").pack(pady=20)

        #Main Layout
        main = tk.Frame(self, bg="white")
        main.pack(fill="both", expand=True, padx=30, pady=150)

        left = tk.Frame(main, bg="white")
        right = tk.Frame(main, bg="white")
        left.pack(side="left", fill="both", expand=True)
        right.pack(side="right", fill="both", expand=True)

        #Image didsplay
        tk.Label(left, text="MNIST image", bg="white", font=("Roboto", 14, "bold")).pack(pady=(0, 10))

        self.imgLabel = tk.Label(left, bg="white", bd=2, relief="solid")
        self.imgLabel.pack(pady=10)


        #Prediction and Status
        tk.Label(right, textvariable=self.activeModelName, bg="white", font=("Roboto", 14, "bold")).pack(pady=(100, 12))

        self.predAI = tk.StringVar(value="AI predicts: -")
        self.statusVar = tk.StringVar(value="Status: -")

        tk.Label(right, textvariable=self.predAI, bg="white", font=("Roboto", 20, "bold")).pack(pady=20)
        tk.Label(right, textvariable=self.statusVar, bg="white", font=("Roboto", 14, "bold")).pack(pady=8)

        #Controls
        controls = tk.Frame(self, bg="white")
        controls.pack(pady=20)

        tk.Button(controls, text="Run MLP", font=("Roboto", 12, "bold"),
                  width=12, height=2, command=self.loadMLP).grid(row=0, column=0, padx=8)

        tk.Button(controls, text="Run CNN", font=("Roboto", 12, "bold"),
                  width=12, height=2, command=self.loadCNN).grid(row=0, column=1, padx=8)

        tk.Button(controls, text="Random", font=("Roboto", 12, "bold"),
                  width=12, height=2, command=self.randomSample).grid(row=0, column=2, padx=8)

        tk.Button(controls, text="Next", font=("Roboto", 12, "bold"),
                  width=12, height=2, command=self.nextSample).grid(row=0, column=3, padx=8)

        tk.Button(controls, text="Trở lại", font=("Roboto", 12, "bold"),
                  width=12, height=2, command=self.controller.DrawMainUserScreen).grid(row=0, column=4, padx=8)

        # show first image
        self.showSample(self.curIdx)

    # Select models
    def loadMLP(self):
        if self.modelMLP is None:
            self.modelMLP = loadModel(self.device, "mlp")
        self.activeModel = self.modelMLP
        self.activeModelName.set(f"Model: {self.activeModel.__class__.__name__} (MLP)")
        self.predictCurrent()

    def loadCNN(self):
        if self.modelCNN is None:
            self.modelCNN = loadModel(self.device, "cnn")
        self.activeModel = self.modelCNN
        self.activeModelName.set(f"Model: {self.activeModel.__class__.__name__} (CNN)")
        self.predictCurrent()

    # Choose sample
    def nextSample(self):
        self.curIdx = (self.curIdx + 1) % len(self.dataset)
        self.showSample(self.curIdx)
        self.predictCurrent()

    def randomSample(self):
        self.curIdx = random.randint(0, len(self.dataset) - 1)
        self.showSample(self.curIdx)
        self.predictCurrent()

    # Img display
    def showSample(self, idx: int):
        imgTensor, label = self.dataset[idx]         
        imgForShow = imgTensor.clone()

        # unnormalize for display
        imgForShow = imgForShow * 0.3081 + 0.1307
        imgForShow = torch.clamp(imgForShow, 0, 1)

        # to PIL
        pil = transforms.ToPILImage()(imgForShow)
        pil = pil.resize((280, 280), Image.NEAREST)

        self._tk_img = ImageTk.PhotoImage(pil)
        self.imgLabel.configure(image=self._tk_img)

    #Get predict
    def predictCurrent(self):
        if self.activeModel is None:
            self.predAI.set("AI predicts: -")
            self.statusVar.set("Status: (chưa chọn model)")
            return

        imgTensor, label = self.dataset[self.curIdx]
        x = imgTensor.unsqueeze(0).to(self.device)    

        with torch.no_grad():
            out = self.activeModel(x)
            probs = F.softmax(out, dim=1)[0]          

        pred = int(torch.argmax(probs).item())

        self.predAI.set(f"AI predicts: {pred}")

        if pred == int(label):
            self.statusVar.set("Status: ĐÚNG")
        else:
            self.statusVar.set("Status: SAI")
