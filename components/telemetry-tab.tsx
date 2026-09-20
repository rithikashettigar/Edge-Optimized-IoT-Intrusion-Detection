"use client"

import { useEffect, useRef } from "react"
import { Grid3x3, TerminalSquare, Eraser } from "lucide-react"
import { Card, Button, Badge } from "./primitives"
import { SectionHeader } from "./section"
import { confusionMatrix } from "@/lib/mock-data"

const { tp, fp, fn, tn } = confusionMatrix
const total = tp + fp + fn + tn
const precision = tp / (tp + fp)
const recall = tp / (tp + fn)
const f1 = (2 * precision * recall) / (precision + recall)
const accuracy = (tp + tn) / total

function Cell({
  label,
  value,
  kind,
}: {
  label: string
  value: number
  kind: "good" | "bad"
}) {
  const pct = ((value / total) * 100).toFixed(2)
  return (
    <div
      className={`flex flex-col items-center justify-center rounded-xl border p-4 text-center transition-colors ${
        kind === "good"
          ? "border-emerald/30 bg-emerald/10 hover:border-emerald/60"
          : "border-rose/30 bg-rose/10 hover:border-rose/60"
      }`}
    >
      <span className={`text-2xl font-semibold ${kind === "good" ? "text-emerald" : "text-rose"}`}>
        {value.toLocaleString()}
      </span>
      <span className="mt-1 text-xs font-medium text-foreground">{label}</span>
      <span className="text-[11px] text-muted-foreground">{pct}%</span>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border-subtle bg-surface-muted/40 px-3 py-2 text-center">
      <p className="text-lg font-semibold text-cyan">{value}</p>
      <p className="text-[11px] text-muted-foreground">{label}</p>
    </div>
  )
}

export function TelemetryTab({ logs, onClear }: { logs: string[]; onClear: () => void }) {
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-5">
      <Card className="p-5 xl:col-span-2">
        <SectionHeader
          icon={Grid3x3}
          title="Confusion Matrix"
          subtitle="Quantized INT8 student · held-out test set"
          accent="emerald"
        />
        <div className="mt-4 grid grid-cols-2 gap-3">
          <Cell label="True Positives" value={tp} kind="good" />
          <Cell label="False Positives" value={fp} kind="bad" />
          <Cell label="False Negatives" value={fn} kind="bad" />
          <Cell label="True Negatives" value={tn} kind="good" />
        </div>
        <div className="mt-4 grid grid-cols-4 gap-2">
          <Metric label="Accuracy" value={`${(accuracy * 100).toFixed(1)}%`} />
          <Metric label="Precision" value={`${(precision * 100).toFixed(1)}%`} />
          <Metric label="Recall" value={`${(recall * 100).toFixed(1)}%`} />
          <Metric label="F1-Score" value={`${(f1 * 100).toFixed(1)}%`} />
        </div>
      </Card>

      <Card className="flex flex-col p-5 xl:col-span-3" hover={false}>
        <SectionHeader
          icon={TerminalSquare}
          title="Live Streaming Logs"
          subtitle="stdout · compression tasks & deployment handshakes"
          accent="cyan"
          action={
            <Button accent="neutral" onClick={onClear} className="px-2.5 py-1.5 text-xs">
              <Eraser className="h-3.5 w-3.5" /> Clear
            </Button>
          }
        />
        <div className="relative mt-4 flex-1 overflow-hidden rounded-xl border border-border-subtle bg-[oklch(0.12_0.02_255)]">
          <div className="flex items-center gap-1.5 border-b border-border-subtle px-4 py-2.5">
            <span className="h-2.5 w-2.5 rounded-full bg-rose/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-amber/70" />
            <span className="h-2.5 w-2.5 rounded-full bg-emerald/70" />
            <span className="ml-2 font-mono text-[11px] text-muted-foreground">edgeids@router:~/ops</span>
            <Badge accent="emerald" className="ml-auto">
              <span className="live-dot inline-block h-1.5 w-1.5 rounded-full bg-emerald text-emerald" />
              live
            </Badge>
          </div>
          <div className="custom-scroll h-[340px] overflow-y-auto p-4 font-mono text-xs leading-relaxed">
            {logs.length === 0 ? (
              <p className="text-muted-foreground">
                <span className="text-emerald">$</span> awaiting task execution — trigger a pipeline step to stream output…
              </p>
            ) : (
              logs.map((line, i) => (
                <div key={i} className="flex gap-2 text-foreground/90">
                  <span className="shrink-0 select-none text-muted-foreground/50">
                    {String(i + 1).padStart(3, "0")}
                  </span>
                  <span className={lineColor(line)}>{line}</span>
                </div>
              ))
            )}
            <div ref={endRef} />
          </div>
        </div>
      </Card>
    </div>
  )
}

function lineColor(line: string) {
  if (line.includes("✓") || line.includes("complete") || line.includes("ready") || line.includes("converged"))
    return "text-emerald"
  if (line.includes("[prune]")) return "text-amber"
  if (line.includes("[distill]")) return "text-emerald/90"
  if (line.includes("[quant]")) return "text-cyan"
  if (line.includes("[deploy]")) return "text-rose/90"
  return "text-foreground/80"
}
