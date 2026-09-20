"use client"

import { ShieldAlert, ShieldCheck, Activity, Crosshair, Gauge, Radio } from "lucide-react"
import { Card, Button, Badge } from "@/components/primitives"
import { SectionHeader } from "@/components/section"
import { attacks, type Attack } from "@/lib/attacks"
import { cn } from "@/lib/utils"

type DetStats = { scanned: number; caught: number; missed: number; lastConf: number }

export function AttackSimulatorTab({
  onLaunch,
  activeAttack,
  stats,
  feed,
}: {
  onLaunch: (a: Attack) => void
  activeAttack: string | null
  stats: DetStats
  feed: string[]
}) {
  const detectionRate = stats.scanned > 0 ? (stats.caught / stats.scanned) * 100 : 0

  return (
    <div className="flex flex-col gap-6">
      <Card hover={false} className="border-rose/20 bg-rose/[0.03] p-4">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald" />
          <div className="text-xs leading-relaxed text-muted-foreground">
            <span className="font-semibold text-foreground">Safe sandbox.</span> No real malware runs here. Each button
            injects a burst of <span className="text-foreground">synthetic network flows</span> carrying the statistical
            signature of a known attack (from labeled datasets like CIC-IDS2017 / Bot-IoT). The quantized INT8 model
            scores every flow in real time so you can watch it detect and block the threat.
          </div>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        {attacks.map((a) => {
          const Icon = a.icon
          const active = activeAttack === a.id
          const disabled = activeAttack !== null && !active
          return (
            <Card key={a.id} className={cn("flex flex-col gap-4 p-5", active && "shadow-[0_0_0_1px_var(--color-rose)]")}>
              <SectionHeader
                icon={Icon}
                title={a.name}
                subtitle={a.family}
                accent={a.accent}
                action={
                  active ? (
                    <Badge accent="rose">
                      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-rose" />
                      Live
                    </Badge>
                  ) : undefined
                }
              />
              <p className="text-xs leading-relaxed text-muted-foreground">{a.desc}</p>
              <div className="rounded-lg border border-border-subtle bg-surface-muted/50 p-3">
                <div className="mb-1 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  <Crosshair className="h-3 w-3" />
                  Signature
                </div>
                <p className="font-mono text-[11px] leading-relaxed text-foreground/80">{a.signature}</p>
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  {a.flows} flows · ports{" "}
                  <span className="font-mono text-foreground/70">{a.ports.slice(0, 3).join(", ")}…</span>
                </span>
                <Badge accent={a.accent}>{a.tag}</Badge>
              </div>
              <Button accent={a.accent} loading={active} disabled={disabled} onClick={() => onLaunch(a)} className="w-full">
                {active ? "Attack in progress…" : (
                  <>
                    <ShieldAlert className="h-4 w-4" />
                    Launch {a.name}
                  </>
                )}
              </Button>
            </Card>
          )
        })}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.1fr_1fr]">
        <Card hover={false} className="p-5">
          <SectionHeader
            icon={Gauge}
            title="Live Detection Metrics"
            subtitle="Updated as the model scores each injected flow"
            accent="emerald"
          />
          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Flows Scanned" value={stats.scanned.toLocaleString()} icon={Activity} accent="cyan" />
            <Stat label="Threats Blocked" value={stats.caught.toLocaleString()} icon={ShieldCheck} accent="emerald" />
            <Stat label="Missed" value={stats.missed.toLocaleString()} icon={ShieldAlert} accent="rose" />
            <Stat
              label="Detection Rate"
              value={`${detectionRate.toFixed(1)}%`}
              icon={Gauge}
              accent={detectionRate >= 95 ? "emerald" : "amber"}
            />
          </div>
          <div className="mt-4 rounded-lg border border-border-subtle bg-surface-muted/40 px-4 py-3">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground">Last flow confidence</span>
              <span className="font-mono font-semibold text-foreground">
                {stats.lastConf > 0 ? stats.lastConf.toFixed(3) : "—"}
              </span>
            </div>
          </div>
        </Card>

        <Card hover={false} className="flex flex-col p-5">
          <SectionHeader icon={Radio} title="Detection Feed" subtitle="Per-flow model verdicts" accent="rose" />
          <div className="mt-4 h-64 overflow-y-auto rounded-lg border border-border-subtle bg-[#05070d] p-3 font-mono text-[11px] leading-relaxed">
            {feed.length === 0 ? (
              <p className="text-muted-foreground/60">Launch an attack to stream live detections…</p>
            ) : (
              feed.map((line, i) => (
                <div
                  key={i}
                  className={cn(
                    "whitespace-pre-wrap",
                    line.includes("BLOCKED")
                      ? "text-emerald"
                      : line.includes("MISSED")
                        ? "text-rose"
                        : line.includes("[sim]")
                          ? "text-amber"
                          : "text-foreground/70",
                  )}
                >
                  {line}
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

function Stat({
  label,
  value,
  icon: Icon,
  accent,
}: {
  label: string
  value: string
  icon: typeof Activity
  accent: "cyan" | "emerald" | "amber" | "rose"
}) {
  const map = {
    cyan: "text-cyan",
    emerald: "text-emerald",
    amber: "text-amber",
    rose: "text-rose",
  }
  return (
    <div className="rounded-lg border border-border-subtle bg-surface-muted/40 p-3">
      <Icon className={cn("mb-1.5 h-4 w-4", map[accent])} />
      <div className="font-mono text-lg font-semibold text-foreground">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
    </div>
  )
}
