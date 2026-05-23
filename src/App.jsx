// src/App.jsx
import React, { useState, useEffect, useRef } from 'react';
import * as ort from 'onnxruntime-web';

export default function App() {
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const [heatLoad, setHeatLoad] = useState(20.0); // Interactive structural scalar Q

  // Domain geometry specs matching our PyTorch optimization setup
  const Nx = 80;
  const Ny = 40;
  const trueK = 0.45;
  const trainedK = 0.48937; // The non-trivial extrapolation value found by our PINN

  // Exact spatial coordinates of the 15 industrial sensors used during training
  const sensors = [
    { x: 0.831, y: 0.701 }, { x: 1.721, y: 0.241 }, { x: 1.328, y: 0.466 },
    { x: 1.117, y: 0.225 }, { x: 0.312, y: 0.424 }, { x: 0.311, y: 0.211 },
    { x: 0.243, y: 0.732 }, { x: 1.733, y: 0.722 }, { x: 1.202, y: 0.605 },
    { x: 1.411, y: 0.114 }, { x: 0.942, y: 0.868 }, { x: 1.634, y: 0.832 },
    { x: 0.423, y: 0.325 }, { x: 1.135, y: 0.514 }, { x: 0.722, y: 0.183 }
  ];

  // 1. Mount the ONNX graph architecture stiching external parameters
  useEffect(() => {
    async function loadModel() {
      try {
        console.log("Fetching ONNX structural model configuration...");
        const responseModel = await fetch('/inverse_thermal_model.onnx');
        const modelBuffer = await responseModel.arrayBuffer();
        
        console.log("Fetching companion parameter weights matrix...");
        const responseData = await fetch('/inverse_thermal_model.onnx.data');
        const dataBuffer = await responseData.arrayBuffer();
        const externalDataArray = new Uint8Array(dataBuffer);

        const runtimeSession = await ort.InferenceSession.create(modelBuffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
          externalData: [
            {
              path: 'inverse_thermal_model.onnx.data',
              data: externalDataArray
            }
          ]
        });

        console.log("Inverse model environment compiled successfully!");
        setSession(runtimeSession);
        setLoading(false);
      } catch (err) {
        console.error("Failed to compile ONNX Runtime Web assembly graph:", err);
      }
    }
    loadModel();
  }, []);

  // 2. Real-time inference evaluation loop parsing continuous coordinate spaces
  useEffect(() => {
    if (!session) return;

    async function evaluateFields() {
      const xCoords = [];
      const yCoords = [];

      // Unpack matrix grids matching structural [0, 2] x [0, 1] parameters
      for (let y = 0; y < Ny; y++) {
        for (let x = 0; x < Nx; x++) {
          const normX = (x / (Nx - 1)) * 2.0;
          const normY = (y / (Ny - 1)) * 1.0;
          xCoords.push(normX);
          yCoords.push(normY);
        }
      }

      // Convert coordinate streams into separate input matrix vectors [3200, 1]
      const xInputTensor = new ort.Tensor('float32', new Float32Array(xCoords), [Nx * Ny, 1]);
      const yInputTensor = new ort.Tensor('float32', new Float32Array(yCoords), [Nx * Ny, 1]);

      // Query dynamic compiled layer names from the session structure automatically
      const inputs = {};
      inputs[session.inputNames[0]] = xInputTensor;
      inputs[session.inputNames[1]] = yInputTensor;

      const outputs = await session.run(inputs);
      const outputKey = Object.keys(outputs)[0];
      const rawTemperatures = outputs[outputKey].data;

      // Scale the continuous field temperature dynamically based on the current UI Heat Load scale
      const calibratedTemperatures = new Float32Array(Nx * Ny);
      const scalingFactor = heatLoad / 20.0; 
      for (let i = 0; i < rawTemperatures.length; i++) {
        calibratedTemperatures[i] = rawTemperatures[i] * scalingFactor;
      }

      renderCanvas(calibratedTemperatures);
    }

    evaluateFields();
  }, [session, heatLoad]);

// 3. Render 2D heat gradients and overlay hardware thermocouple boundaries
  const renderCanvas = (temperatures) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cellW = canvas.width / Nx;
    const cellH = canvas.height / Ny;

    // FIX: Anchor the color scale normalization to a fixed high-end ceiling 
    // instead of self-scaling dynamically. This lets the visualization react to the slider.
    const maxReferenceTemp = 150.0; // Captures peak heat at max wattage limits

    // Step A: Draw the global element thermal gradient map
    for (let y = 0; y < Ny; y++) {
      for (let x = 0; x < Nx; x++) {
        const index = y * Nx + x;
        
        // Clamp the value between 0 and 1 so the pixel color math never breaks
        let normalizedVal = temperatures[index] / maxReferenceTemp;
        if (normalizedVal > 1.0) normalizedVal = 1.0;
        if (normalizedVal < 0.0) normalizedVal = 0.0;

        // Scientific "Thermal Iron" color mapping profile
        const r = Math.floor(normalizedVal * 255);
        const g = Math.floor(Math.pow(normalizedVal, 2.5) * 160);
        const b = Math.floor(Math.pow(1 - normalizedVal, 2) * 100 + (normalizedVal > 0.8 ? (normalizedVal - 0.8) * 500 : 0));

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x * cellW, (Ny - 1 - y) * cellH, cellW + 0.3, cellH + 0.3);
      }
    }

    // Step B: Overlay the physical noisy industrial thermocouple sensor icons (Leave unchanged)
    sensors.forEach((sensor, index) => {
      const canvasX = (sensor.x / 2.0) * canvas.width;
      const canvasY = canvas.height - (sensor.y / 1.0) * canvas.height;

      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 7, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(56, 189, 248, 0.4)';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
    });
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', height: '100vh', justifyContent: 'center', alignItems: 'center', backgroundColor: '#090d16', color: '#38bdf8', fontFamily: 'monospace' }}>
        <h2>INITIALIZING INVERSE PHYSICAL EXPERIMENT CORE...</h2>
      </div>
    );
  }

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#090d16', color: '#f1f5f9', fontFamily: 'sans-serif', padding: '2rem' }}>
      <header style={{ borderBottom: '1px solid #1e293b', paddingBottom: '1.5rem', marginBottom: '2.5rem' }}>
        <h1 style={{ margin: '0.25rem 0 0.5rem 0', fontSize: '2.25rem', fontWeight: '800' }}>Heat Exchanger Parameter Estimation</h1>
<p style={{ margin: 0, color: '#64748b', fontSize: '1rem' }}>Extracting hidden material properties (k) from sparse, noisy thermocouple telemetry fields via continuous gradient checking.</p>      </header>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: '2.5rem', maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ backgroundColor: '#111827', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#94a3b8' }}>Steady-State Thermal Field Distribution</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#38bdf8' }}></span>
                <span style={{ fontSize: '0.8rem', fontFamily: 'monospace', color: '#38bdf8' }}>15 ACTIVE SENSOR NODES OVERLAYED</span>
              </div>
            </div>
            <div style={{ borderRadius: '8px', overflow: 'hidden', border: '2px solid #1f2937' }}>
              <canvas ref={canvasRef} width={800} height={400} style={{ display: 'block', width: '100%' }} />
            </div>
          </div>

          <div style={{ backgroundColor: '#111827', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
<label style={{ fontWeight: '700', color: '#f1f5f9' }}>Volumetric Internal Heat Generation (Q)</label>              <span style={{ color: '#38bdf8', fontFamily: 'monospace', fontWeight: 'bold' }}>{heatLoad.toFixed(1)} kW/m³</span>
            </div>
            <input 
              type="range" min="5.0" max="40.0" step="0.5" value={heatLoad}
              onChange={(e) => setHeatLoad(parseFloat(e.target.value))}
              style={{ width: '100%', cursor: 'pointer', accentColor: '#38bdf8' }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ backgroundColor: '#111827', border: '1px solid #1e293b', borderRadius: '12px', padding: '1.75rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', borderBottom: '1px solid #1e293b', paddingBottom: '0.75rem' }}>Diagnostic Metrics</h3>
            
            <div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#64748b', display: 'block' }}>INITIAL GUESS PARAMETER (k_init)</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ef4444', fontFamily: 'monospace' }}>1.5000 <span style={{ fontSize: '0.9rem', color: '#64748b' }}>W/m·K</span></div>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#64748b', display: 'block' }}>TRUE HIDDEN MATERIAL CONSTANT (k_true)</span>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#10b981', fontFamily: 'monospace' }}>0.4500 <span style={{ fontSize: '0.9rem', color: '#64748b' }}>W/m·K</span></div>
            </div>

            <div style={{ backgroundColor: '#090d16', border: '1px solid #1e293b', borderRadius: '8px', padding: '1rem' }}>
              <span style={{ fontSize: '0.75rem', fontFamily: 'monospace', color: '#38bdf8', display: 'block' }}>PINN EXTRAPOLATED VALUE (k_est)</span>
              <div style={{ fontSize: '2rem', fontWeight: '900', color: '#38bdf8', fontFamily: 'monospace' }}>
                {trainedK.toFixed(5)}
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px dashed #1e293b', marginTop: '0.5rem', paddingTop: '0.5rem', fontSize: '0.8rem' }}>
                <span style={{ color: '#64748b' }}>Calculated Error:</span>
                <span style={{ color: '#38bdf8', fontFamily: 'monospace', fontWeight: 'bold' }}>
                  {(((trainedK - trueK) / trueK) * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}