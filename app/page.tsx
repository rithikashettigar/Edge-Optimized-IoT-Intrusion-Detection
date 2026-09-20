"use client"

import { useEffect, useRef, useState } from "react"
import { LayoutDashboard, SlidersHorizontal, Radar, PackageCheck } from "lucide-react"
import { TopBar } from "@/components/top-bar"
import { OverviewTab } from "@/components/overview-tab"
import { PipelineTab } from "@/components/pipeline-tab"
import { TelemetryTab } from "@/components/telemetry-tab"
import { ExportPanel } from "@/components/export-panel"
import { initialAnomalyData, makeAnomalyPoint } from "@/lib/mock-data"
import { cn } from "@/lib/utils"

const tabs = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "pipeline", label: "Pipeline Control", icon: SlidersHorizontal },
  { id: "telemetry", label: "Telemetry & Benchmarks", icon: Radar },
  { id: "export", label: "Export & Deploy", icon: PackageCheck },
] as const

type TabId = (typeof tabs)[number]["id"]

export default function Page() {
  const [tab, setTab] = useState<TabId>("overview")
  const [logs, setLogs] = useState<string[]>([])
  const [anomalyData, setAnomalyData] = useState(initialAnomalyData)
  const tRef = useRef(initialAnomalyData.length)

  const log = (line: string) => {
    const stamp = new Date().toLocaleTimeString("en-US", { hour12: false })
    setLogs((prev) => [...prev.slice(-200), `${stamp}  ${line}`])
  }

  // Live anomaly stream
  useEffect(() => {
    const id = setInterval(() => {
      setAnomalyData((prev) => {
        const next = [...prev.slice(1), makeAnomalyPoint(tRef.current)]
        tRef.current += 1
        return next
      })
    }, 2200)
    return () => clearInterval(id)
  }, [])

  const latestAttack = anomalyData[anomalyData.length - 1]?.attack ?? 0
  const threatLevel = latestAttack > 60 ? "Critical" : latestAttack > 25 ? "Elevated" : "Low"
  const flow = (12.4 + (anomalyData[anomalyData.length - 1]?.normal ?? 0) / 20).toFixed(1)

  return (
    <div className="min-h-screen">
      <TopBar telemetryFlow={`${flow}k/s`} compressionStage="Quantized INT8" threatLevel={threatLevel} />

      <main className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6">
        <nav className="mb-6 flex flex-wrap gap-1.5 rounded-2xl border border-border-subtle bg-surface/50 p-1.5 backdrop-blur">
          {tabs.map((t) => {
            const Icon = t.icon
            const active = tab === t.id
            return (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={cn(
                  "flex flex-1 items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 sm:flex-none",
                  active
                    ? "bg-cyan/15 text-cyan shadow-[0_0_0_1px_var(--color-cyan)]"
                    : "text-muted-foreground hover:bg-surface-muted/60 hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="whitespace-nowrap">{t.label}</span>
              </button>
            )
          })}
        </nav>

        <div key={tab} style={{ animation: "fadeUp 0.3s ease-out" }}>
          {tab === "overview" && <OverviewTab anomalyData={anomalyData} />}
          {tab === "pipeline" && <PipelineTab log={log} />}
          {tab === "telemetry" && <TelemetryTab logs={logs} onClear={() => setLogs([])} />}
          {tab === "export" && <ExportPanel log={log} />}
        </div>
      </main>

      <footer className="mx-auto max-w-[1600px] px-4 py-6 text-center text-[11px] text-muted-foreground sm:px-6">
        EdgeIDS-Ops · Model Compression &amp; Deployment Control Center · ARMv8 Edge Runtime
      </footer>

      <style>{`@keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }`}</style>
    </div>
  )
}
