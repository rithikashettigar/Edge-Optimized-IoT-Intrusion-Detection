"use client"

import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"
import { cn } from "@/lib/utils"

export function SectionHeader({
  icon: Icon,
  title,
  subtitle,
  accent = "cyan",
  action,
}: {
  icon: LucideIcon
  title: string
  subtitle?: string
  accent?: "cyan" | "emerald" | "amber" | "rose"
  action?: ReactNode
}) {
  const map = {
    cyan: "border-cyan/40 bg-cyan/10 text-cyan",
    emerald: "border-emerald/40 bg-emerald/10 text-emerald",
    amber: "border-amber/40 bg-amber/10 text-amber",
    rose: "border-rose/40 bg-rose/10 text-rose",
  }
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex items-center gap-3">
        <div className={cn("flex h-9 w-9 items-center justify-center rounded-lg border", map[accent])}>
          <Icon className="h-4.5 w-4.5" style={{ height: "1.125rem", width: "1.125rem" }} />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          {subtitle && <p className="text-xs text-muted-foreground">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  )
}
