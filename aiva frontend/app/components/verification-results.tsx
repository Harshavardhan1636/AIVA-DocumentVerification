"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { CheckCircle, AlertTriangle, Copy, ExternalLink } from "lucide-react"

interface VerificationData {
  documentType: string
  extractedText: string
  fraudScore: number
  blockchainHash: string
  isVerified: boolean
  confidence: number
  details: {
    verification_steps?: { name: string; status: "success" | "warning" | "error" }[]
  }
}

interface VerificationResultsProps {
  data: VerificationData
  onReset: () => void
}

export default function VerificationResults({ data, onReset }: VerificationResultsProps) {
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const getVerificationStatus = () => {
    if (data.isVerified && data.confidence > 90) {
      return { status: "verified", color: "green", icon: CheckCircle }
    } else if (data.confidence > 70) {
      return { status: "warning", color: "yellow", icon: AlertTriangle }
    } else {
      return { status: "failed", color: "red", icon: AlertTriangle }
    }
  }

  const { status, color, icon: StatusIcon } = getVerificationStatus()

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Verification Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <StatusIcon className={`h-5 w-5 text-${color}-600`} />
            Verification Results
          </CardTitle>
          <CardDescription>Document analysis complete</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Status Badge */}
          <div className="flex items-center gap-2">
            <Badge
              variant={status === "verified" ? "default" : "destructive"}
              className={`${
                status === "verified"
                  ? "bg-green-100 text-green-800 border-green-200"
                  : status === "warning"
                    ? "bg-yellow-100 text-yellow-800 border-yellow-200"
                    : "bg-red-100 text-red-800 border-red-200"
              }`}
            >
              {status === "verified" ? "VERIFIED" : status === "warning" ? "NEEDS REVIEW" : "FAILED"}
            </Badge>
            <span className="text-sm text-gray-500">{data.documentType}</span>
          </div>

          {/* Confidence Score */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Confidence Score</span>
              <span className="text-sm font-bold">{data.confidence}%</span>
            </div>
            <Progress value={data.confidence} className="h-2" />
          </div>

          {/* Fraud Score */}
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium">Fraud Risk</span>
              <span className="text-sm font-bold text-red-600">{data.fraudScore}%</span>
            </div>
            <Progress value={data.fraudScore} className="h-2" />
          </div>

          {/* Blockchain Hash */}
          <div>
            <label className="text-sm font-medium text-gray-700">Blockchain Hash</label>
            <div className="flex items-center gap-2 mt-1">
              <code className="text-xs bg-gray-100 px-2 py-1 rounded flex-1 truncate">{data.blockchainHash}</code>
              <Button size="sm" variant="outline" onClick={() => copyToClipboard(data.blockchainHash)}>
                <Copy className="h-3 w-3" />
              </Button>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4">
            <Button onClick={onReset} variant="outline" className="flex-1 bg-transparent">
              Verify Another
            </Button>
            <Button variant="outline" size="sm">
              <ExternalLink className="h-4 w-4 mr-2" />
              View on Explorer
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Extracted Data */}
      <Card>
        <CardHeader>
          <CardTitle>Extracted Information</CardTitle>
          <CardDescription>Data extracted from document using OCR</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-gray-50 p-4 rounded-lg font-mono text-sm space-y-1">
            {data.extractedText.split('\n').map((line, index) => {
              const parts = line.split(':');
              const key = parts[0];
              const value = parts.slice(1).join(':').trim();
              if (parts.length > 1) {
                return (
                  <div key={index} className="flex">
                    <span className="font-semibold w-24">{key}:</span>
                    <span className="flex-1 truncate">{value}</span>
                  </div>
                );
              }
              return <div key={index} className="font-bold">{line}</div>;
            })}
          </div>

          <div className="mt-4 space-y-2">
            <h4 className="font-medium text-sm">Verification Details:</h4>
            <ul className="text-sm text-gray-600 space-y-1">
              {data.details?.verification_steps?.map((step, index) => (
                <li key={index} className="flex items-center">
                  {step.status === 'success' ? (
                    <CheckCircle className="h-4 w-4 text-green-500 mr-2" />
                  ) : (
                    <AlertTriangle className="h-4 w-4 text-yellow-500 mr-2" />
                  )}
                  <span>{step.name}</span>
                </li>
              ))}
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
