"use client"

import {
  Line,
  LineChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts"
import { latencyData } from "@/lib/mock-data"
import { ChartTooltip } from "./chart-tooltip"

export function LatencyChart() {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={latencyData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
        <XAxis
          dataKey="pruning"
          stroke="var(--color-muted-foreground)"
          fontSize={11}
          tickLine={false}
          axisLine={false}
          unit="%"
        />
        <YAxis
          stroke="var(--color-muted-foreground)"
          fontSize={11}
          tickLine={false}
          axisLine={false}
          unit="ms"
        />
        <Tooltip content={<ChartTooltip unit="ms" />} cursor={{ stroke: "var(--color-cyan)", strokeOpacity: 0.2 }} />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
        <Line
          type="monotone"
          dataKey="edge"
          name="Edge Router"
          stroke="var(--color-cyan)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="router"
          name="Raspberry Pi 4"
          stroke="var(--color-amber)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
        <Line
          type="monotone"
          dataKey="jetson"
          name="Jetson Nano"
          stroke="var(--color-emerald)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
