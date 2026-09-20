import type { Metadata, Viewport } from "next"
import type { ReactNode } from "react"
import { ToastProvider } from "@/components/toast"
import "./globals.css"

export const metadata: Metadata = {
  title: "EdgeIDS-Ops | Model Compression & Deployment Control Center",
  description:
    "Enterprise-grade edge intrusion detection operations console for knowledge distillation, structured pruning, quantization, and edge deployment.",
  generator: "v0.app",
}

export const viewport: Viewport = {
  themeColor: "#0b1220",
  colorScheme: "dark",
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="antialiased">
        <ToastProvider>{children}</ToastProvider>
      </body>
    </html>
  )
}
