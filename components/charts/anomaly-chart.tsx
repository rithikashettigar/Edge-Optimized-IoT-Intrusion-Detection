"use client"

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { ChartTooltip } from "./chart-tooltip"

export function AnomalyChart({ data }: { data: { label: string; normal: number; attack: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <defs>
          <linearGradient id="normalGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--color-cyan)" stopOpacity={0.5} />
            <stop offset="95%" stopColor="var(--color-cyan)" stopOpacity={0.02} />
          </linearGradient>
          <linearGradient id="attackGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="var(--color-rose)" stopOpacity={0.6} />
            <stop offset="95%" stopColor="var(--color-rose)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
        <XAxis dataKey="label" stroke="var(--color-muted-foreground)" fontSize={10} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis stroke="var(--color-muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
        <Tooltip content={<ChartTooltip />} cursor={{ stroke: "var(--color-rose)", strokeOpacity: 0.2 }} />
        <Area
          type="monotone"
          dataKey="normal"
          name="Normal Flow"
          stroke="var(--color-cyan)"
          strokeWidth={2}
          fill="url(#normalGrad)"
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="attack"
          name="Attack Signatures"
          stroke="var(--color-rose)"
          strokeWidth={2}
          fill="url(#attackGrad)"
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  )
}
