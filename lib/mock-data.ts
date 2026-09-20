export const latencyData = [
  { pruning: 0, edge: 3.9, router: 4.6, jetson: 2.7 },
  { pruning: 10, edge: 3.4, router: 4.0, jetson: 2.4 },
  { pruning: 20, edge: 2.9, router: 3.5, jetson: 2.1 },
  { pruning: 30, edge: 2.5, router: 3.0, jetson: 1.8 },
  { pruning: 40, edge: 2.1, router: 2.6, jetson: 1.5 },
  { pruning: 50, edge: 1.7, router: 2.1, jetson: 1.3 },
  { pruning: 60, edge: 1.4, router: 1.8, jetson: 1.1 },
  { pruning: 70, edge: 1.1, router: 1.4, jetson: 0.95 },
  { pruning: 80, edge: 0.9, router: 1.1, jetson: 0.85 },
  { pruning: 90, edge: 0.8, router: 0.95, jetson: 0.8 },
]

export const memoryData = [
  { model: "Teacher MLP", size: 185, fill: "var(--color-cyan-dim)" },
  { model: "Distilled Student", size: 42, fill: "var(--color-cyan)" },
  { model: "Pruned Student", size: 8.6, fill: "var(--color-emerald)" },
  { model: "Quantized INT8", size: 1.4, fill: "var(--color-amber)" },
]

export function makeAnomalyPoint(t: number) {
  const normal = 40 + Math.sin(t / 3) * 12 + Math.random() * 10
  const attackBase = Math.random() > 0.82 ? 30 + Math.random() * 55 : Math.random() * 8
  return {
    t,
    label: `T-${t}`,
    normal: Math.max(0, Math.round(normal)),
    attack: Math.max(0, Math.round(attackBase)),
  }
}

export const initialAnomalyData = Array.from({ length: 24 }, (_, i) => makeAnomalyPoint(i))

export const confusionMatrix = {
  tp: 18432,
  fp: 214,
  fn: 176,
  tn: 92847,
}

export const hardwareTargets = [
  { id: "rpi4", name: "Raspberry Pi 4", arch: "ARMv8 Cortex-A72", ram: "4 GB" },
  { id: "edge-router", name: "Edge Router", arch: "ARMv8 Class", ram: "2 GB" },
  { id: "jetson", name: "Jetson Nano", arch: "Maxwell 128-core", ram: "4 GB" },
]

export const pruningMethods = [
  "Incoming Weight Norms",
  "L1 Structured",
  "L2 Structured",
  "Taylor Expansion",
  "Random Baseline",
]
