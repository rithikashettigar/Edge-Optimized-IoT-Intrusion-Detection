"use client"

import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { memoryData } from "@/lib/mock-data"
import { ChartTooltip } from "./chart-tooltip"

export function MemoryChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={memoryData} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
        <XAxis
          dataKey="model"
          stroke="var(--color-muted-foreground)"
          fontSize={10}
          tickLine={false}
          axisLine={false}
          interval={0}
        />
        <YAxis
          stroke="var(--color-muted-foreground)"
          fontSize={11}
          tickLine={false}
          axisLine={false}
          unit="MB"
        />
        <Tooltip content={<ChartTooltip unit=" MB" />} cursor={{ fill: "var(--color-surface-muted)", fillOpacity: 0.4 }} />
        <Bar dataKey="size" name="Footprint" radius={[6, 6, 0, 0]}>
          {memoryData.map((entry) => (
            <Cell key={entry.model} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
