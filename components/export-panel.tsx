"use client"

import { useState } from "react"
import { Download, FileText, FileSpreadsheet, UploadCloud, Boxes, Rocket } from "lucide-react"
import { Card, Button, Badge } from "./primitives"
import { SectionHeader } from "./section"
import { useToast } from "./toast"
import { hardwareTargets } from "@/lib/mock-data"

type LogFn = (line: string) => void

const edgeNodes = [
  { id: "node-01", name: "gateway-rpi4-01", status: "ready" },
  { id: "node-02", name: "router-armv8-02", status: "ready" },
  { id: "node-03", name: "jetson-edge-03", status: "syncing" },
]

export function ExportPanel({ log }: { log: LogFn }) {
  const toast = useToast()
  const [pushing, setPushing] = useState<string | null>(null)

  const download = (filename: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
    toast({ title: "Download started", description: filename, kind: "success" })
  }

  const pushConfig = (nodeName: string) => {
    setPushing(nodeName)
    log(`[deploy] opening secure channel → ${nodeName}`)
    setTimeout(() => log(`[deploy] transferring model_int8.tflite (1.4MB) → ${nodeName}`), 500)
    setTimeout(() => log(`[deploy] handshake ACK · checksum verified · ${nodeName}`), 1200)
    setTimeout(() => {
      log(`[deploy] ${nodeName} now serving compressed IDS model ✓`)
      setPushing(null)
      toast({ title: "Deployment complete", description: `${nodeName} updated`, kind: "success" })
    }, 1800)
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card className="flex flex-col gap-4 p-5">
        <SectionHeader icon={Boxes} title="Model Artifacts" subtitle="Compiled & quantized weights" accent="cyan" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <div className="flex flex-col gap-3 rounded-xl border border-border-subtle bg-surface-muted/40 p-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm text-foreground">model.onnx</span>
              <Badge accent="cyan">8.6 MB</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Pruned student · ONNX opset 17</p>
            <Button accent="cyan" onClick={() => download("model.onnx", "ONNX_MODEL_BINARY_PLACEHOLDER\nopset=17\nparams=pruned_student")}>
              <Download className="h-4 w-4" /> Download .onnx
            </Button>
          </div>
          <div className="flex flex-col gap-3 rounded-xl border border-border-subtle bg-surface-muted/40 p-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-sm text-foreground">model_int8.tflite</span>
              <Badge accent="amber">1.4 MB</Badge>
            </div>
            <p className="text-xs text-muted-foreground">Quantized INT8 · edge-ready</p>
            <Button accent="amber" onClick={() => download("model_int8.tflite", "TFLITE_INT8_BINARY_PLACEHOLDER\nquant=int8\ntarget=armv8")}>
              <Download className="h-4 w-4" /> Download .tflite
            </Button>
          </div>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Button
            accent="emerald"
            onClick={() =>
              download(
                "benchmark_report.csv",
                "stage,size_mb,f1,latency_ms\nteacher,185,0.9930,3.9\ndistilled,42,0.9940,2.1\npruned,8.6,0.9930,1.4\nquant_int8,1.4,0.9920,0.8\n",
              )
            }
          >
            <FileSpreadsheet className="h-4 w-4" /> Export CSV
          </Button>
          <Button
            accent="neutral"
            onClick={() => download("benchmark_report.pdf", "%PDF-1.4 EdgeIDS-Ops Benchmark Report (placeholder)")}
          >
            <FileText className="h-4 w-4" /> Export PDF
          </Button>
        </div>
      </Card>

      <Card className="flex flex-col gap-4 p-5">
        <SectionHeader icon={Rocket} title="Edge Node Deployment" subtitle="Push configs to registered nodes" accent="emerald" />
        <div className="flex flex-col gap-3">
          {edgeNodes.map((node) => (
            <div
              key={node.id}
              className="flex items-center justify-between gap-3 rounded-xl border border-border-subtle bg-surface-muted/40 p-3.5"
            >
              <div className="flex items-center gap-3">
                <span
                  className={`inline-block h-2 w-2 rounded-full ${
                    node.status === "ready" ? "bg-emerald live-dot text-emerald" : "bg-amber text-amber"
                  }`}
                />
                <div>
                  <p className="font-mono text-sm text-foreground">{node.name}</p>
                  <p className="text-[11px] capitalize text-muted-foreground">{node.status}</p>
                </div>
              </div>
              <Button
                accent="emerald"
                loading={pushing === node.name}
                disabled={node.status !== "ready" && pushing !== node.name}
                onClick={() => pushConfig(node.name)}
                className="px-3 py-1.5 text-xs"
              >
                <UploadCloud className="h-3.5 w-3.5" /> Push
              </Button>
            </div>
          ))}
        </div>
        <p className="mt-auto text-[11px] leading-relaxed text-muted-foreground">
          Targets: {hardwareTargets.map((t) => t.name).join(" · ")}. Deployments stream signed artifacts over mTLS and
          verify checksums before activation.
        </p>
      </Card>
    </div>
  )
}
