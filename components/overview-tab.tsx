"use client"

import { TrendingDown, MemoryStick, Target, Timer, ArrowDownRight, ArrowUpRight, LineChart as LineIcon, BarChart3, Waves } from "lucide-react"
import { Card } from "./primitives"
import { SectionHeader } from "./section"
import { LatencyChart } from "./charts/latency-chart"
import { MemoryChart } from "./charts/memory-chart"
import { AnomalyChart } from "./charts/anomaly-chart"

const kpis = [
  {
    icon: TrendingDown,
    label: "Model Size Reduction",
    value: "99.2%",
    sub: "185 MB → 1.4 MB",
    trend: "down",
    accent: "cyan" as const,
  },
  {
    icon: MemoryStick,
    label: "Active Weight Memory",
    value: "1.4 MB",
    sub: "vs 185 MB teacher",
    trend: "down",
    accent: "emerald" as const,
  },
  {
    icon: Target,
    label: "F1-Score Retention",
    value: "99.4%",
    sub: "+0.1% over baseline",
    trend: "up",
    accent: "amber" as const,
  },
  {
    icon: Timer,
    label: "Inference Latency",
    value: "0.8ms",
    sub: "single-thread ARMv8",
    trend: "down",
    accent: "cyan" as const,
  },
]

const accentText = {
  cyan: "text-cyan",
  emerald: "text-emerald",
  amber: "text-amber",
}

export function OverviewTab({ anomalyData }: { anomalyData: { label: string; normal: number; attack: number }[] }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {kpis.map((k) => {
          const Icon = k.icon
          const TrendIcon = k.trend === "up" ? ArrowUpRight : ArrowDownRight
          return (
            <Card key={k.label} className="p-5">
              <div className="flex items-start justify-between">
                <div className={`flex h-10 w-10 items-center justify-center rounded-lg bg-surface-muted ${accentText[k.accent]}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <TrendIcon className={`h-4 w-4 ${k.trend === "up" ? "text-emerald" : "text-cyan"}`} />
              </div>
              <p className="mt-4 text-3xl font-semibold tracking-tight text-foreground">{k.value}</p>
              <p className="mt-1 text-sm font-medium text-foreground">{k.label}</p>
              <p className="text-xs text-muted-foreground">{k.sub}</p>
            </Card>
          )
        })}
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <Card className="p-5">
          <SectionHeader
            icon={LineIcon}
            title="Inference Latency vs. Pruning %"
            subtitle="Per deployment layer"
            accent="cyan"
          />
          <div className="mt-4">
            <LatencyChart />
          </div>
        </Card>

        <Card className="p-5">
          <SectionHeader
            icon={BarChart3}
            title="Model Memory Footprint"
            subtitle="Compression pipeline stages (MB)"
            accent="amber"
          />
          <div className="mt-4">
            <MemoryChart />
          </div>
        </Card>
      </div>

      <Card className="p-5">
        <SectionHeader
          icon={Waves}
          title="Live Network Flow Anomaly Detection"
          subtitle="Real-time normal traffic vs. detected attack signatures"
          accent="rose"
          action={
            <span className="flex items-center gap-2 text-xs text-muted-foreground">
              <span className="live-dot inline-block h-2 w-2 rounded-full bg-rose text-rose" />
              streaming
            </span>
          }
        />
        <div className="mt-4">
          <AnomalyChart data={anomalyData} />
        </div>
      </Card>
    </div>
  )
}
