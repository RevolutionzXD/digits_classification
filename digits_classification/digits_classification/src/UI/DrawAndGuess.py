import tkinter as tk
import torch
from src.infer import loadModel
import io
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from src.utils.image_utils import ChangePilToMnistTensor

class BrushSlider(tk.Canvas):
    def __init__(self, master, from_=30, to=50, initial=50, command=None, **kwargs):
        self.width = kwargs.pop("width", 220)
        self.height = kwargs.pop("height", 30)
        super().__init__(master, width=self.width, height=self.height,
                         bg=kwargs.pop("bg", "white"), highlightthickness=0)

        self.from_ = from_
        self.to = to
        self.value = initial
        self.command = command

        self.pad = 15
        self.centerY = self.height // 2
        self.radius = 10

        self.trackIdx = None
        self.knobIdx = None
        self.DrawAll()

        self.bind("<Button-1>", self.OnClick)
        self.bind("<B1-Motion>", self.OnDrag)

    def DrawAll(self):
        self.delete("all")
        leftTrackPos = self.pad
        rightTrackPos = self.width - self.pad

        self.trackIdx = self.create_line(
            leftTrackPos, self.centerY, rightTrackPos, self.centerY,
            width=12,
            capstyle=tk.ROUND,
            fill="gray"
        )

        currentPosX = self.ChangeValueToX(self.value)
        self.knobIdx = self.create_oval(
            currentPosX - self.radius, self.centerY - self.radius,
            currentPosX + self.radius, self.centerY + self.radius,
            fill="black",
            outline=""
        )

    def ChangeValueToX(self, v):
        leftTrackPos = self.pad
        rightTrackPos = self.width - self.pad
        t = (v - self.from_) / (self.to - self.from_)
        return leftTrackPos + t * (rightTrackPos - leftTrackPos)

    def ChangeXToValue(self, x):
        leftTrackPos = self.pad
        rightTrackPos = self.width - self.pad
        x = max(leftTrackPos, min(rightTrackPos, x))
        t = (x - leftTrackPos) / (rightTrackPos - leftTrackPos)
        v = self.from_ + t * (self.to - self.from_)
        return int(round(v))

    def SetValue(self, v):
        self.value = max(self.from_, min(self.to, v))
        currentPosX = self.ChangeValueToX(self.value)
        self.coords(
            self.knobIdx,
            currentPosX - self.radius, self.centerY - self.radius,
            currentPosX + self.radius, self.centerY + self.radius
        )
        if self.command:
            self.command(self.value)

    def OnClick(self, event):
        self.SetValue(self.ChangeXToValue(event.x))

    def OnDrag(self, event):
        self.SetValue(self.ChangeXToValue(event.x))

    def get(self):
        return self.value


class DrawAndGuessScreen(tk.Frame):
    def __init__(self, userScreen, controller):
        super().__init__(userScreen, bg="white")
        self.controller = controller
        self.pack(fill="both", expand=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelMLP = None
        self.modelCNN = None

       

        tk.Label(self,
                 text="Draw Your Digit",
                 font=("Roboto", 30, "bold"),
                 bg="white").pack(pady=40)

        # ===== Canvas =====
        self.canvasWidth = 560
        self.canvasHeight = 560

        self.pilImage = Image.new("L", (self.canvasWidth, self.canvasHeight), 0)
        self.pilDraw = ImageDraw.Draw(self.pilImage)

        self.canvas = tk.Canvas(self, width=self.canvasWidth,
                                height=self.canvasHeight, bg="black",
                                highlightthickness=1, highlightbackground="black")
        self.canvas.place(relx=0.25, rely=0.5, anchor="center")

        self.lastX = None
        self.lastY = None

        self.canvas.bind("<Button-1>", self.StartDraw)
        self.canvas.bind("<B1-Motion>", self.Draw)
        self.canvas.bind("<ButtonRelease-1>", self.StopDraw)

        # ===== Slider độ dày nét vẽ (kiểu volume) =====
        self.brushSize = 30   # giá trị hiện tại

        self.rightPannel = tk.Frame(self, bg="white")  # lưu lại để destroy
        self.rightPannel.place(relx=0.6, rely=0.3, anchor ="n")

        tk.Label(self.rightPannel, text="Brush Size", bg="white"
                 , font=("Roboto", 12, "bold")).pack(pady=(0, 5))

        self.brushSlider = BrushSlider(
            self.rightPannel,
            from_=30,
            to=50,
            initial=self.brushSize,
            command=self.OnBrushChange,
            bg="white",
            width=220,
            height=30
        )
        self.brushSlider.pack()

        # ===== Khung nút bên phải (lưu lại để còn destroy) =====

        clearButton = tk.Button(self.rightPannel, text="Xóa ô vẽ",
                                font=("roboto", 15, "bold"),
                                width=10, height=3,
                                command=self.ClearCanvas)
        clearButton.pack(pady=10)

        guessButton = tk.Button(self.rightPannel, text="Let me guess",
                                font=("roboto", 15, "bold"),
                                width=10, height=3,
                                command=self.OnGuessClicked)
        guessButton.pack(pady=10)

        self.modelChoiceFrame = None


        self.resultVar = tk.StringVar(value="Draw and click Let me guess")
        self.resultLabel = tk.Label(
            self,
            textvariable=self.resultVar,
            bg="white",
            font=("roboto", 18, "bold"),
            fg="blue"
        )
        self.resultLabel.place(relx=0.6, rely=0.75, anchor="center")

    def OnBrushChange(self, value: int):
        self.brushSize = value

    #Let me guess
    def OnGuessClicked(self):

        #Destroy right button
        if self.rightPannel is not None:
            self.rightPannel.destroy()
            self.rightPannel = None

        #Create model button
        self.modelChoiceFrame = tk.Frame(self, bg="white")
        self.modelChoiceFrame.place(relx=0.6, rely=0.35, anchor="n")

        buttonMLP = tk.Button(self.modelChoiceFrame,
                            text="MLP Module",
                            font=("roboto", 15, "bold"),
                            width=12, height=2,
                            command=lambda: self.ChooseModel("mlp"))
        buttonMLP.pack(pady=5)

        buttonCNN = tk.Button(self.modelChoiceFrame,
                            text="CNN Module",
                            font=("roboto", 15, "bold"),
                            width=12, height=2,
                            command=lambda: self.ChooseModel("cnn"))
        buttonCNN.pack(pady=5)

        buttonBack = tk.Button(
            self.modelChoiceFrame,
            text="Trở lại",
            font=("roboto", 14, "bold"),
            width=12, height=2,
            command=self.OnBackClicked
        )
        buttonBack.pack(pady=(15, 5))

    def ChooseModel(self, mode: str):
        mode = mode.lower()
        pilImg = self.pilImage.copy()
        # pilImg.save("debug_canvas.png")
        # print("Saved debug_canvas.png")

        if mode == "mlp":
            if self.modelMLP is None:
                self.modelMLP = loadModel(self.device, "mlp")
            model = self.modelMLP

        elif mode == "cnn":
            if self.modelCNN is None:
                self.modelCNN = loadModel(self.device, "cnn")
            model = self.modelCNN

        else:
            print("Mode không hợp lệ:", mode)
            return

        #print("Loaded:", model.__class__.__name__)


        #Dùng LẠI preprocess từ demo.py
        inputTensor = ChangePilToMnistTensor(pilImg, self.device)

        #Predict
        with torch.no_grad():
            out = model(inputTensor)
            probs = torch.softmax(out, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0, pred].item() * 100

        #Hiển thị
        self.resultVar.set(f"AI đoán: {pred} ({conf:.1f}%)")

    #Draw function 
    def StartDraw(self, event):
        self.lastX = event.x
        self.lastY = event.y

    def Draw(self, event):
        if self.lastX is not None and self.lastY is not None:
            # Vẽ trên Tkinter canvas
            self.canvas.create_line(
                self.lastX, self.lastY, event.x, event.y,
                width=self.brushSize,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True
            )

            # Vẽ song song lên PIL 
            self.pilDraw.line(
                [self.lastX, self.lastY, event.x, event.y],
                fill=255,
                width=self.brushSize
            )

        self.lastX = event.x
        self.lastY = event.y


    def StopDraw(self, event):
        self.lastX = None
        self.lastY = None

    def ClearCanvas(self):
        self.canvas.delete("all")
        self.pilImage = Image.new("L", (self.canvasWidth, self.canvasHeight), 0)
        self.pilDraw = ImageDraw.Draw(self.pilImage)
    def OnBackClicked(self):
        self.controller.DrawAndGuessMenu()
