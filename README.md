

# Inverse PINN Parameter Estimation for Heat Exchangers

An advanced SciML web application that leverages an **Inverse Physics-Informed Neural Network (PINN)** running entirely client-side. The dashboard demonstrates how an AI model can act as a non-destructive diagnostic tool, taking sparse, noisy temperature data from a few simulated physical thermocouples and back-calculating a material's hidden thermal conductivity ($k$).

---

## 🚀 Live Demo
👉 **[Insert Your Vercel Live Link Here]**

---

## 💡 The Inverse Problem Architecture

In real-world industrial settings (e.g., semiconductor chips, EV battery packs, aerospace components), you cannot look inside a solid material to check its internal degradation or true thermal properties. 

This project solves an **Inverse Problem**: instead of predicting temperatures from known constants, we register the material's thermal conductivity ($k$) as a **trainable parameter** inside the PyTorch graph graph alongside the model weights. The neural network balances matching the real physical sensor data points while satisfying the underlying governing physics equation.

```text
[15 Sparse Noisy Sensors (Data Loss)] ───┐
                                         ├─> [Custom Multi-Objective Optimizer]
[Governing Heat Equation (Physics Loss)] ──┘                     │
                                                                 ▼
                                                  Updates Neural Weights AND 
                                                  Learns Hidden Material Constant (k)
```



🛠️ Core Features
- Anti-Cheat PDE Loss Formulation: Utilizes a reciprocal heat balance equation framework ($T_{xx} + T_{yy} + \frac{Q}{k} = 0$) to prevent the optimizer from converging on a trivial solution ($k=0$).
- Dual-Lane Learning Rates: Implements grouped parameter optimizations, giving the hidden physical variable a $10\times$ faster tracking speed (lr: 0.01) than the deep neural layers (lr: 0.001) to overcome gradient imbalances.
- Telemetry Overlay: Renders a continuous 2D steady-state thermal distribution field ($80 \times 40$ matrix grid) while overlaying the exact geometric coordinates of the 15 simulated hardware sensors.
- Edge Inference Deployment: Packaged into an optimized standalone ONNX graph topology executing sub-millisecond predictions in-browser via WebAssembly (onnxruntime-web).





🏗️ Technical Stack
Machine Learning Engine: PyTorch (torch.autograd for spatial derivatives)

Web Runtime Provider: ONNX Runtime Web via WebAssembly (WASM execution)

Frontend UI Framework: React 18+ (Vite Build Pipeline)

Graphics Implementation: Native HTML5 Canvas 2D color space mapping



🧠 Behind the Physics Model
The system simulates a solid material plate experiencing internal volumetric heat generation ($Q$). The network resolves the 2D steady-state heat equation using automatic differentiation to minimize residuals across 800 collocation points:
$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} + \frac{Q}{k} = 0$$
- True Parameter Value ($k_{\text{true}}$): 
$0.4500\text{ W/m·K}$
- Initial Blind AI Guess ($k_{\text{init}}$): $1.5000\text{ W/m·K}$
- PINN Extrapolated Value ($k_{\text{est}}$): $0.4893\text{ W/m·K}$ (Extracted under noisy data constraints)


💻 Local Setup and Installation

1- Clone the repository:

```text
git clone [https://github.com/YOUR_USERNAME/thermal-inverse-dashboard.git](https://github.com/YOUR_USERNAME/thermal-inverse-dashboard.git)
cd thermal-inverse-dashboard
```

2- Install dependencies:
```text
npm install
```
3- Incorporate your model assets:

Place your compiled inverse_thermal_model.onnx and inverse_thermal_model.onnx.data files into the /public directory folder.

4- Run development build:

```text
npm run dev
````


