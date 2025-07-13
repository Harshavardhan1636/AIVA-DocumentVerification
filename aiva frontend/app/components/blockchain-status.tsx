"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { BitcoinIcon as Blockchain, Wifi, Clock } from "lucide-react"

export default function BlockchainStatus() {
  const [blockNumber, setBlockNumber] = useState(18234567)
  const [gasPrice, setGasPrice] = useState(23)

  useEffect(() => {
    // Simulate real-time blockchain data updates
    const interval = setInterval(() => {
      setBlockNumber((prev) => prev + Math.floor(Math.random() * 3))
      setGasPrice((prev) => prev + Math.floor(Math.random() * 5) - 2)
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Blockchain className="h-5 w-5" />
          Blockchain Status
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm">Network</span>
          <Badge variant="outline" className="text-blue-600 border-blue-600">
            <Wifi className="h-3 w-3 mr-1" />
            Sepolia Testnet
          </Badge>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm">Latest Block</span>
          <span className="text-sm font-mono">#{blockNumber.toLocaleString()}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm">Gas Price</span>
          <span className="text-sm font-mono">{gasPrice} gwei</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm">Response Time</span>
          <span className="text-sm text-green-600 font-medium">
            <Clock className="h-3 w-3 inline mr-1" />
            1.2s
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
