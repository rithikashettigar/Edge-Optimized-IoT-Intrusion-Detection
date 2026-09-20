"use client"

export function ChartTooltip({
  active,
  payload,
  label,
  unit = "",
}: {
  active?: boolean
  payload?: any[]
  label?: string | number
  unit?: string
}) {
  if (!active || !payload || payload.length === 0) return null
  return (
    <div className="glass rounded-lg border px-3 py-2 text-xs shadow-xl">
      {label !== undefined && (
        <p className="mb-1.5 font-medium text-muted-foreground">
          {typeof label === "number" ? `Pruning ${label}%` : label}
        </p>
      )}
      <div className="flex flex-col gap-1">
        {payload.map((entry, i) => (
          <div key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5">
              <span
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: entry.color || entry.payload?.fill }}
              />
              <span className="text-muted-foreground">{entry.name}</span>
            </span>
            <span className="font-mono font-medium text-foreground">
              {entry.value}
              {unit}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
