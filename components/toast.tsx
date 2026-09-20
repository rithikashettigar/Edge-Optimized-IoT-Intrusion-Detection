"use client"

import { createContext, useCallback, useContext, useState, type ReactNode } from "react"
import { CheckCircle2, AlertTriangle, Info, X } from "lucide-react"
import { cn } from "@/lib/utils"

type ToastKind = "success" | "warning" | "info"
type Toast = { id: number; title: string; description?: string; kind: ToastKind }

const ToastContext = createContext<(t: Omit<Toast, "id">) => void>(() => {})

export function useToast() {
  return useContext(ToastContext)
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])

  const push = useCallback((t: Omit<Toast, "id">) => {
    const id = Date.now() + Math.random()
    setToasts((prev) => [...prev, { ...t, id }])
    setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), 4000)
  }, [])

  const dismiss = (id: number) => setToasts((prev) => prev.filter((x) => x.id !== id))

  const icon = {
    success: <CheckCircle2 className="h-4 w-4 text-emerald" />,
    warning: <AlertTriangle className="h-4 w-4 text-amber" />,
    info: <Info className="h-4 w-4 text-cyan" />,
  }
  const ring = {
    success: "border-emerald/40",
    warning: "border-amber/40",
    info: "border-cyan/40",
  }

  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-full max-w-sm flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={cn(
              "glass pointer-events-auto flex items-start gap-3 rounded-xl border px-4 py-3 shadow-lg",
              "animate-[slideIn_0.25s_ease-out]",
              ring[t.kind],
            )}
            style={{ animation: "fadeSlide 0.25s ease-out" }}
          >
            <div className="mt-0.5">{icon[t.kind]}</div>
            <div className="flex-1">
              <p className="text-sm font-medium text-foreground">{t.title}</p>
              {t.description && <p className="mt-0.5 text-xs text-muted-foreground">{t.description}</p>}
            </div>
            <button
              onClick={() => dismiss(t.id)}
              className="text-muted-foreground transition-colors hover:text-foreground"
              aria-label="Dismiss notification"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
      <style>{`@keyframes fadeSlide { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }`}</style>
    </ToastContext.Provider>
  )
}
