"use client"

import { useEffect, useRef, useState } from "react"
import { LayoutDashboard, SlidersHorizontal, Radar, PackageCheck, ShieldAlert } from "lucide-react"
import { TopBar } from "@/components/top-bar"
import { OverviewTab } from "@/components/overview-tab"
import { PipelineTab } from "@/components/pipeline-tab"
import { TelemetryTab } from "@/components/telemetry-tab"
import { ExportPanel } from "@/components/export-panel"
import { AttackSimulatorTab } from "@/components/attack-simulator-tab"
import { initialAnomalyData, makeAnomalyPoint } from "@/lib/mock-data"
import { randomFlowSource, type Attack } from "@/lib/attacks"
import { useToast } from "@/components/toast"
import { cn } from "@/lib/utils"

const tabs = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "pipeline", label: "Pipeline Control", icon: SlidersHorizontal },
  { id: "simulator", label: "Attack Simulator", icon: ShieldAlert },
  { id: "telemetry", label: "Telemetry & Benchmarks", icon: Radar },
  { id: "export", label: "Export & Deploy", icon: PackageCheck },
] as const

type TabId = (typeof tabs)[number]["id"]

export default function Page() {
  const [tab, setTab] = useState<TabId>("overview")
  const [logs, setLogs] = useState<string[]>([])
  const [anomalyData, setAnomalyData] = useState(initialAnomalyData)
  const tRef = useRef(initialAnomalyData.length)
  const toast = useToast()

  // Attack simulator state
  const [activeAttack, setActiveAttack] = useState<string | null>(null)
  const [detStats, setDetStats] = useState({ scanned: 0, caught: 0, missed: 0, lastConf: 0 })
  const [feed, setFeed] = useState<string[]>([])
  const burstRef = useRef(0)

  const log = (line: string) => {
    const stamp = new Date().toLocaleTimeString("en-US", { hour12: false })
    setLogs((prev) => [...prev.slice(-200), `${stamp}  ${line}`])
  }

  const feedLog = (line: string) => setFeed((prev) => [...prev.slice(-120), line])

  const launchAttack = (a: Attack) => {
    if (activeAttack) return
    setActiveAttack(a.id)
    burstRef.current = Math.ceil(a.flows / 2)
    toast({ title: `${a.name} detected`, description: "Injecting synthetic attack flows", kind: "warning" })
    log(`[sim] ⚠ ${a.name} launched — injecting ${a.flows} synthetic ${a.tag} flows`)
    feedLog(`[sim] ⚠ ${a.name} — ${a.family}`)

    let i = 0
    const id = setInterval(() => {
      const conf = a.confMin + Math.random() * (a.confMax - a.confMin)
      const caught = Math.random() < a.recall
      const src = randomFlowSource()
      const port = a.ports[Math.floor(Math.random() * a.ports.length)]
      const verdict = caught ? "BLOCKED" : "MISSED"
      const line = `[detect] ${a.tag} src=${src}:${port} conf=${conf.toFixed(3)} → ${verdict}`
      feedLog(line)
      log(line)
      setDetStats((s) => ({
        scanned: s.scanned + 1,
        caught: s.caught + (caught ? 1 : 0),
        missed: s.missed + (caught ? 0 : 1),
        lastConf: conf,
      }))
      i += 1
      if (i >= a.flows) {
        clearInterval(id)
        setActiveAttack(null)
        const msg = `[sim] ✓ ${a.name} contained — ${a.flows} flows scored, threat neutralized`
        log(msg)
        feedLog(msg)
        toast({ title: `${a.name} contained`, description: "Threat neutralized by edge model", kind: "success" })
      }
    }, 300)
  }

  // Live anomaly stream (attack bursts elevate the signal)
  useEffect(() => {
    const id = setInterval(() => {
      setAnomalyData((prev) => {
        let point
        if (burstRef.current > 0) {
          burstRef.current -= 1
          point = {
            t: tRef.current,
            label: `T-${tRef.current}`,
            normal: Math.round(28 + Math.random() * 12),
            attack: Math.round(72 + Math.random() * 26),
          }
        } else {
          point = makeAnomalyPoint(tRef.current)
        }
        tRef.current += 1
        return [...prev.slice(1), point]
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
          {tab === "simulator" && (
            <AttackSimulatorTab onLaunch={launchAttack} activeAttack={activeAttack} stats={detStats} feed={feed} />
          )}
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
