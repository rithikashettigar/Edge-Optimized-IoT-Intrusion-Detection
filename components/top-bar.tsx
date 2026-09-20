"use client"

import { ShieldHalf, Activity, Cpu, Radio } from "lucide-react"
import { Badge } from "./primitives"

type ThreatLevel = "Low" | "Elevated" | "Critical"

export function TopBar({
  telemetryFlow,
  compressionStage,
  threatLevel,
}: {
  telemetryFlow: string
  compressionStage: string
  threatLevel: ThreatLevel
}) {
  const threatAccent = threatLevel === "Critical" ? "rose" : threatLevel === "Elevated" ? "amber" : "emerald"

  return (
    <header className="glass sticky top-0 z-30 rounded-none border-x-0 border-t-0 px-4 py-3 sm:px-6">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan/40 bg-cyan/10">
            <ShieldHalf className="h-5 w-5 text-cyan" />
          </div>
          <div className="leading-tight">
            <h1 className="text-sm font-semibold tracking-tight text-foreground sm:text-base">
              EdgeIDS-Ops
            </h1>
            <p className="text-[11px] text-muted-foreground">Model Compression &amp; Deployment Control Center</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="live-dot inline-block h-2 w-2 rounded-full bg-emerald text-emerald" />
          <span className="text-xs text-muted-foreground">
            Edge Node: <span className="font-medium text-emerald">Online</span> · Router_Class_ARMv8
          </span>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <Badge accent="cyan">
            <Activity className="h-3.5 w-3.5" />
            Telemetry Flow: {telemetryFlow}
          </Badge>
          <Badge accent="amber">
            <Cpu className="h-3.5 w-3.5" />
            Stage: {compressionStage}
          </Badge>
          <Badge accent={threatAccent}>
            <Radio className="h-3.5 w-3.5" />
            Threat: {threatLevel}
          </Badge>
        </div>
      </div>
    </header>
  )
}
