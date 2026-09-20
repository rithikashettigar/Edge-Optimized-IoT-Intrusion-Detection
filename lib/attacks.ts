import type { LucideIcon } from "lucide-react"
import { Bot, Waves, ScanSearch } from "lucide-react"

export type Attack = {
  id: string
  name: string
  tag: string
  family: string
  icon: LucideIcon
  accent: "rose" | "amber" | "cyan"
  desc: string
  signature: string
  flows: number // number of synthetic flows in the burst
  recall: number // model detection rate (0-1)
  confMin: number
  confMax: number
  ports: number[]
}

export const attacks: Attack[] = [
  {
    id: "botnet",
    name: "Mirai Botnet",
    tag: "BOTNET",
    family: "IoT C2 / Mirai",
    icon: Bot,
    accent: "rose",
    desc: "Compromised devices beaconing to a command-and-control server with periodic keep-alive traffic.",
    signature: "Repeated outbound TCP to fixed C2 host · low-entropy payload · fan-out to peers",
    flows: 14,
    recall: 0.985,
    confMin: 0.93,
    confMax: 0.999,
    ports: [23, 2323, 48101, 7547],
  },
  {
    id: "ddos",
    name: "DDoS Flood",
    tag: "DDOS",
    family: "Volumetric / SYN Flood",
    icon: Waves,
    accent: "amber",
    desc: "High-rate SYN flood from spoofed sources attempting to exhaust the target's connection table.",
    signature: "Extreme packet rate · half-open connections · uniform tiny flow duration",
    flows: 20,
    recall: 0.994,
    confMin: 0.9,
    confMax: 0.998,
    ports: [80, 443, 53, 8080],
  },
  {
    id: "portscan",
    name: "Port Scan",
    tag: "PORTSCAN",
    family: "Reconnaissance / Nmap",
    icon: ScanSearch,
    accent: "cyan",
    desc: "Sequential probing across many destination ports to map open services on the target host.",
    signature: "Single src → many dst ports · SYN with no ACK · sub-millisecond inter-flow gaps",
    flows: 16,
    recall: 0.968,
    confMin: 0.82,
    confMax: 0.97,
    ports: [21, 22, 25, 110, 139, 445, 3306, 3389, 5432, 8443],
  },
]

export function randomFlowSource() {
  const o = () => Math.floor(Math.random() * 254) + 1
  return `192.168.${Math.floor(Math.random() * 4)}.${o()}`
}
