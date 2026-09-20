"use client"

import { cn } from "@/lib/utils"
import type { ReactNode } from "react"

type Accent = "cyan" | "emerald" | "amber" | "rose" | "neutral"

const accentBtn: Record<Accent, string> = {
  cyan: "bg-cyan/15 text-cyan border-cyan/40 hover:bg-cyan/25 hover:border-cyan/70",
  emerald: "bg-emerald/15 text-emerald border-emerald/40 hover:bg-emerald/25 hover:border-emerald/70",
  amber: "bg-amber/15 text-amber border-amber/40 hover:bg-amber/25 hover:border-amber/70",
  rose: "bg-rose/15 text-rose border-rose/40 hover:bg-rose/25 hover:border-rose/70",
  neutral: "bg-surface-muted text-foreground border-border-subtle hover:bg-surface-muted/70",
}

export function Button({
  children,
  onClick,
  accent = "cyan",
  disabled,
  loading,
  className,
  type = "button",
}: {
  children: ReactNode
  onClick?: () => void
  accent?: Accent
  disabled?: boolean
  loading?: boolean
  className?: string
  type?: "button" | "submit"
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed active:scale-[0.98]",
        accentBtn[accent],
        className,
      )}
    >
      {loading && (
        <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
      )}
      {children}
    </button>
  )
}

export function Progress({ value, accent = "cyan" }: { value: number; accent?: Accent }) {
  const bar: Record<Accent, string> = {
    cyan: "bg-cyan",
    emerald: "bg-emerald",
    amber: "bg-amber",
    rose: "bg-rose",
    neutral: "bg-muted-foreground",
  }
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-muted">
      <div
        className={cn("h-full rounded-full transition-all duration-300 ease-out", bar[accent])}
        style={{ width: `${Math.min(100, Math.max(0, value))}%` }}
      />
    </div>
  )
}

export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  accent = "cyan",
}: {
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  accent?: Accent
}) {
  const pct = ((value - min) / (max - min)) * 100
  const track: Record<Accent, string> = {
    cyan: "var(--color-cyan)",
    emerald: "var(--color-emerald)",
    amber: "var(--color-amber)",
    rose: "var(--color-rose)",
    neutral: "var(--color-muted-foreground)",
  }
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="h-2 w-full cursor-pointer appearance-none rounded-full outline-none [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-white [&::-webkit-slider-thumb]:shadow-md [&::-webkit-slider-thumb]:transition-transform [&::-webkit-slider-thumb]:hover:scale-110"
      style={{
        background: `linear-gradient(to right, ${track[accent]} 0%, ${track[accent]} ${pct}%, var(--color-surface-muted) ${pct}%, var(--color-surface-muted) 100%)`,
      }}
    />
  )
}

export function Toggle({
  checked,
  onChange,
  accent = "cyan",
}: {
  checked: boolean
  onChange: (v: boolean) => void
  accent?: Accent
}) {
  const on: Record<Accent, string> = {
    cyan: "bg-cyan",
    emerald: "bg-emerald",
    amber: "bg-amber",
    rose: "bg-rose",
    neutral: "bg-muted-foreground",
  }
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative h-6 w-11 shrink-0 rounded-full border border-white/10 transition-colors duration-200",
        checked ? on[accent] : "bg-surface-muted",
      )}
    >
      <span
        className={cn(
          "absolute top-0.5 h-4.5 w-4.5 rounded-full bg-white shadow transition-transform duration-200",
          checked ? "translate-x-5" : "translate-x-0.5",
        )}
        style={{ height: "1.125rem", width: "1.125rem" }}
      />
    </button>
  )
}

export function Select({
  value,
  options,
  onChange,
  className,
}: {
  value: string
  options: string[]
  onChange: (v: string) => void
  className?: string
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={cn(
        "w-full rounded-lg border border-border-subtle bg-surface-muted px-3 py-2 text-sm text-foreground outline-none transition-colors focus:border-cyan/60",
        className,
      )}
    >
      {options.map((o) => (
        <option key={o} value={o} className="bg-surface">
          {o}
        </option>
      ))}
    </select>
  )
}

export function Card({
  children,
  className,
  hover = true,
}: {
  children: ReactNode
  className?: string
  hover?: boolean
}) {
  return (
    <div className={cn("glass rounded-2xl", hover && "glass-hover", className)}>{children}</div>
  )
}

export function Badge({
  children,
  accent = "neutral",
  className,
}: {
  children: ReactNode
  accent?: Accent
  className?: string
}) {
  const map: Record<Accent, string> = {
    cyan: "bg-cyan/12 text-cyan border-cyan/30",
    emerald: "bg-emerald/12 text-emerald border-emerald/30",
    amber: "bg-amber/12 text-amber border-amber/30",
    rose: "bg-rose/12 text-rose border-rose/30",
    neutral: "bg-surface-muted text-muted-foreground border-border-subtle",
  }
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium",
        map[accent],
        className,
      )}
    >
      {children}
    </span>
  )
}
