"use client"

import { useState } from "react"
import { Database, Brain, Scissors, Binary, Play, Check, Cpu } from "lucide-react"
import { Card, Button, Progress, Slider, Toggle, Select, Badge } from "./primitives"
import { SectionHeader } from "./section"
import { useToast } from "./toast"
import { hardwareTargets, pruningMethods } from "@/lib/mock-data"

type LogFn = (line: string) => void

function useRunner() {
  const [progress, setProgress] = useState(0)
  const [running, setRunning] = useState(false)
  const [done, setDone] = useState(false)

  const run = (steps: string[], log: LogFn, onEach?: (i: number) => void) => {
    if (running) return
    setRunning(true)
    setDone(false)
    setProgress(0)
    let i = 0
    const total = steps.length
    const tick = () => {
      if (i < total) {
        log(steps[i])
        onEach?.(i)
        i += 1
        setProgress(Math.round((i / total) * 100))
        setTimeout(tick, 550 + Math.random() * 350)
      } else {
        setRunning(false)
        setDone(true)
      }
    }
    tick()
  }

  return { progress, running, done, run }
}

function StepShell({
  step,
  icon: Icon,
  title,
  subtitle,
  accent,
  done,
  children,
}: {
  step: number
  icon: typeof Database
  title: string
  subtitle: string
  accent: "cyan" | "emerald" | "amber" | "rose"
  done: boolean
  children: React.ReactNode
}) {
  return (
    <Card className="flex flex-col gap-4 p-5">
      <SectionHeader
        icon={Icon}
        title={`Step ${step} · ${title}`}
        subtitle={subtitle}
        accent={accent}
        action={
          done ? (
            <Badge accent="emerald">
              <Check className="h-3.5 w-3.5" /> Complete
            </Badge>
          ) : (
            <Badge accent="neutral">Idle</Badge>
          )
        }
      />
      {children}
    </Card>
  )
}

export function PipelineTab({ log }: { log: LogFn }) {
  const toast = useToast()

  const prep = useRunner()
  const distill = useRunner()
  const prune = useRunner()
  const quant = useRunner()

  // Step 2 config
  const [temperature, setTemperature] = useState(4.0)
  const [alpha, setAlpha] = useState(0.7)
  const [softLogits, setSoftLogits] = useState(true)

  // Step 3 config
  const [pruningMethod, setPruningMethod] = useState(pruningMethods[0])
  const [sparsity, setSparsity] = useState(50)

  // Step 4 config
  const [quantMode, setQuantMode] = useState<"static" | "dynamic">("static")
  const [target, setTarget] = useState(hardwareTargets[0].id)

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      {/* Step 1 */}
      <StepShell step={1} icon={Database} title="Data Preprocessing" subtitle="Feature scaling & leakage removal" accent="cyan" done={prep.done}>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Removes temporally-leaked features, deduplicates flow records, and applies robust scaling to the CIC-IDS
          telemetry corpus.
        </p>
        <Progress value={prep.progress} accent="cyan" />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">{prep.progress}%</span>
          <Button
            accent="cyan"
            loading={prep.running}
            onClick={() => {
              toast({ title: "Preprocessing started", description: "Cleaning telemetry corpus", kind: "info" })
              prep.run(
                [
                  "[prep] loading 2.83M flow records...",
                  "[prep] dropping leaked columns: Flow ID, Timestamp, Src IP",
                  "[prep] deduplicating → removed 41,204 rows",
                  "[prep] RobustScaler fit on 78 features",
                  "[prep] train/val/test split 70/15/15 complete",
                ],
                log,
              )
            }}
          >
            <Play className="h-4 w-4" /> {prep.done ? "Re-run Cleaning" : "Run Cleaning"}
          </Button>
        </div>
      </StepShell>

      {/* Step 2 */}
      <StepShell step={2} icon={Brain} title="Knowledge Distillation" subtitle="Teacher → Student transfer" accent="emerald" done={distill.done}>
        <div className="flex flex-col gap-4">
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Temperature (T)</span>
              <span className="font-mono text-emerald">{temperature.toFixed(1)}</span>
            </div>
            <Slider value={temperature} min={1} max={10} step={0.5} onChange={setTemperature} accent="emerald" />
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Alpha (soft/hard balance)</span>
              <span className="font-mono text-emerald">{alpha.toFixed(2)}</span>
            </div>
            <Slider value={alpha} min={0} max={1} step={0.05} onChange={setAlpha} accent="emerald" />
          </div>
          <div className="flex items-center justify-between rounded-lg border border-border-subtle bg-surface-muted/50 px-3 py-2">
            <span className="text-xs text-muted-foreground">
              Logit mode: <span className="font-medium text-foreground">{softLogits ? "Soft" : "Hard"}</span>
            </span>
            <Toggle checked={softLogits} onChange={setSoftLogits} accent="emerald" />
          </div>
        </div>
        <Progress value={distill.progress} accent="emerald" />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">{distill.progress}%</span>
          <Button
            accent="emerald"
            loading={distill.running}
            onClick={() => {
              toast({ title: "Distillation training started", description: `T=${temperature.toFixed(1)} · α=${alpha.toFixed(2)}`, kind: "info" })
              distill.run(
                [
                  `[distill] init teacher MLP (185MB) + student (42MB)`,
                  `[distill] KD loss = α·CE + (1-α)·KL(T=${temperature.toFixed(1)})`,
                  "[distill] epoch 1/5  loss=0.4182  val_f1=0.981",
                  "[distill] epoch 3/5  loss=0.1934  val_f1=0.991",
                  "[distill] epoch 5/5  loss=0.0921  val_f1=0.994",
                  "[distill] student converged → checkpoint saved",
                ],
                log,
              )
            }}
          >
            <Play className="h-4 w-4" /> Start Training
          </Button>
        </div>
      </StepShell>

      {/* Step 3 */}
      <StepShell step={3} icon={Scissors} title="Structured Pruning" subtitle="Channel & neuron removal" accent="amber" done={prune.done}>
        <div className="flex flex-col gap-4">
          <div>
            <label className="mb-1.5 block text-xs text-muted-foreground">Pruning Method</label>
            <Select value={pruningMethod} options={pruningMethods} onChange={setPruningMethod} />
          </div>
          <div>
            <div className="mb-1.5 flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Target Sparsity</span>
              <span className="font-mono text-amber">{sparsity}%</span>
            </div>
            <Slider value={sparsity} min={10} max={90} step={10} onChange={setSparsity} accent="amber" />
          </div>
        </div>
        <Progress value={prune.progress} accent="amber" />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">{prune.progress}%</span>
          <Button
            accent="amber"
            loading={prune.running}
            onClick={() => {
              toast({ title: "Pruning engine started", description: `${pruningMethod} @ ${sparsity}%`, kind: "info" })
              prune.run(
                [
                  `[prune] method=${pruningMethod}`,
                  `[prune] computing importance scores over 6 layers`,
                  `[prune] removing ${sparsity}% of channels → 8.6MB`,
                  "[prune] fine-tuning 2 epochs to recover accuracy",
                  "[prune] post-prune val_f1=0.993  ✓",
                ],
                log,
              )
            }}
          >
            <Play className="h-4 w-4" /> Execute Pruning
          </Button>
        </div>
      </StepShell>

      {/* Step 4 */}
      <StepShell step={4} icon={Binary} title="Post-Training Quantization" subtitle="INT8 conversion & targeting" accent="cyan" done={quant.done}>
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-2">
            {(["static", "dynamic"] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setQuantMode(mode)}
                className={`rounded-lg border px-3 py-2 text-xs font-medium capitalize transition-colors ${
                  quantMode === mode
                    ? "border-cyan/60 bg-cyan/15 text-cyan"
                    : "border-border-subtle bg-surface-muted/40 text-muted-foreground hover:text-foreground"
                }`}
              >
                {mode} INT8
              </button>
            ))}
          </div>
          <div>
            <label className="mb-1.5 block text-xs text-muted-foreground">Hardware Target</label>
            <Select
              value={hardwareTargets.find((t) => t.id === target)?.name ?? ""}
              options={hardwareTargets.map((t) => t.name)}
              onChange={(name) => setTarget(hardwareTargets.find((t) => t.name === name)?.id ?? target)}
            />
            <p className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
              <Cpu className="h-3 w-3" />
              {hardwareTargets.find((t) => t.id === target)?.arch} · {hardwareTargets.find((t) => t.id === target)?.ram}
            </p>
          </div>
        </div>
        <Progress value={quant.progress} accent="cyan" />
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-muted-foreground">{quant.progress}%</span>
          <Button
            accent="cyan"
            loading={quant.running}
            onClick={() => {
              const hw = hardwareTargets.find((t) => t.id === target)?.name
              toast({ title: "Quantization started", description: `${quantMode} INT8 → ${hw}`, kind: "info" })
              quant.run(
                [
                  `[quant] mode=${quantMode} INT8  target=${hw}`,
                  "[quant] calibrating with 512 representative batches",
                  "[quant] fusing conv+bn+relu ops",
                  "[quant] weights 8.6MB → 1.4MB (INT8)",
                  "[quant] accuracy delta: -0.001 f1  ✓",
                  "[quant] artifact ready: model_int8.tflite",
                ],
                log,
              )
            }}
          >
            <Play className="h-4 w-4" /> Quantize &amp; Compile
          </Button>
        </div>
      </StepShell>
    </div>
  )
}
