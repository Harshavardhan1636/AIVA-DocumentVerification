"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Camera, Upload, Shield, CheckCircle, BitcoinIcon as Blockchain, Brain } from "lucide-react"
import DocumentScanner from "./components/document-scanner"
import VerificationResults from "./components/verification-results"
import BlockchainStatus from "./components/blockchain-status"

type VerificationStep = "upload" | "scanning" | "ai-analysis" | "blockchain" | "complete"

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

export default function AIVADocumentVerification() {
  const [currentStep, setCurrentStep] = useState<VerificationStep>("upload")
  const [verificationData, setVerificationData] = useState<VerificationData | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [systemStatus, setSystemStatus] = useState<any>(null)
  const [stats, setStats] = useState<any>(null)

  const steps = [
    { id: "upload", label: "Document Upload", icon: Upload },
    { id: "scanning", label: "Vision Processing", icon: Camera },
    { id: "ai-analysis", label: "AI Analysis", icon: Brain },
    { id: "blockchain", label: "Blockchain Verification", icon: Blockchain },
    { id: "complete", label: "Complete", icon: CheckCircle },
  ]

  // Fetch system status and stats on component mount
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        // Fetch system status
        const statusResponse = await fetch('/api/status')
        if (statusResponse.ok) {
          const statusData = await statusResponse.json()
          setSystemStatus(statusData)
        } else {
          console.warn('Failed to fetch system status:', statusResponse.status)
        }

        // Fetch system stats
        const statsResponse = await fetch('/api/stats')
        if (statsResponse.ok) {
          const statsData = await statsResponse.json()
          setStats(statsData)
        } else {
          console.warn('Failed to fetch system stats:', statsResponse.status)
        }
      } catch (error) {
        console.error('Failed to fetch system info:', error)
        // Don't throw error, just log it and continue
      }
    }

    fetchSystemInfo()
  }, [])

  const handleDocumentUpload = async (file: File) => {
    setIsProcessing(true)
    setCurrentStep("scanning")
    setProgress(20)

    try {
      // Create FormData for file upload
      const formData = new FormData()
      formData.append("document", file)

      // Call our API endpoint which will forward to Python backend
      const response = await fetch("/api/verify", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.error || "Verification failed")
      }

      const result = await response.json()
      
      if (result.success) {
        setVerificationData(result.data)
        setCurrentStep("complete")
        setProgress(100)
      } else {
        throw new Error(result.error || "Verification failed")
      }
    } catch (error) {
      console.error("Verification failed:", error)
      // You might want to show an error toast here
    } finally {
      setIsProcessing(false)
    }
  }

  const resetVerification = () => {
    setCurrentStep("upload")
    setVerificationData(null)
    setProgress(0)
    setIsProcessing(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Shield className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold text-gray-900">AIVA</h1>
          </div>
          <p className="text-xl text-gray-600">AI-Powered Document Verification System</p>
          <p className="text-sm text-gray-500 mt-2">Secure • Transparent • Instant</p>
        </div>

        {/* Progress Steps */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            {steps.map((step, index) => {
              const Icon = step.icon
              const isActive = steps.findIndex((s) => s.id === currentStep) >= index
              const isComplete = steps.findIndex((s) => s.id === currentStep) > index

              return (
                <div key={step.id} className="flex flex-col items-center">
                  <div
                    className={`w-12 h-12 rounded-full flex items-center justify-center mb-2 ${
                      isComplete
                        ? "bg-green-500 text-white"
                        : isActive
                          ? "bg-blue-500 text-white"
                          : "bg-gray-200 text-gray-400"
                    }`}
                  >
                    <Icon className="h-6 w-6" />
                  </div>
                  <span className={`text-xs text-center ${isActive ? "text-blue-600 font-medium" : "text-gray-500"}`}>
                    {step.label}
                  </span>
                </div>
              )
            })}
          </div>
          {isProcessing && <Progress value={progress} className="w-full" />}
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Document Scanner */}
          <div className="lg:col-span-2">
            <DocumentScanner
              onDocumentUpload={handleDocumentUpload}
              isProcessing={isProcessing}
              currentStep={currentStep}
            />
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* System Status */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5" />
                  System Status
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {systemStatus ? (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Vision Module (M1)</span>
                      <Badge variant="outline" className={`${systemStatus.vision_module?.status === 'online' ? 'text-green-600 border-green-600' : 'text-red-600 border-red-600'}`}>
                        <CheckCircle className="h-3 w-3 mr-1" />
                        {systemStatus.vision_module?.status || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Blockchain Module (M2)</span>
                      <Badge variant="outline" className={`${systemStatus.blockchain_module?.status === 'connected' ? 'text-green-600 border-green-600' : 'text-red-600 border-red-600'}`}>
                        <CheckCircle className="h-3 w-3 mr-1" />
                        {systemStatus.blockchain_module?.status || 'Unknown'}
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">AI Module (M3)</span>
                      <Badge variant="outline" className={`${systemStatus.ai_module?.status === 'ready' ? 'text-green-600 border-green-600' : 'text-red-600 border-red-600'}`}>
                        <CheckCircle className="h-3 w-3 mr-1" />
                        {systemStatus.ai_module?.status || 'Unknown'}
                      </Badge>
                    </div>
                  </>
                ) : (
                  <div className="text-center text-gray-500">Loading system status...</div>
                )}
              </CardContent>
            </Card>

            {/* Blockchain Status */}
            <BlockchainStatus />

            {/* Quick Stats */}
            <Card>
              <CardHeader>
                <CardTitle>Today's Verifications</CardTitle>
              </CardHeader>
              <CardContent>
                {stats ? (
                  <>
                    <div className="text-3xl font-bold text-blue-600">{stats.total_verifications?.toLocaleString() || '0'}</div>
                    <p className="text-sm text-gray-500">Documents processed</p>
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Success Rate</span>
                        <span className="font-medium">{stats.success_rate || '0'}%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Avg. Processing Time</span>
                        <span className="font-medium">{stats.avg_processing_time || '0'}s</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Today's Verifications</span>
                        <span className="font-medium">{stats.today_verifications || '0'}</span>
                      </div>
                    </div>
                  </>
                ) : (
                  <div className="text-center text-gray-500">Loading statistics...</div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Verification Results */}
        {verificationData && (
          <div className="mt-8">
            <VerificationResults data={verificationData} onReset={resetVerification} />
          </div>
        )}
      </div>
    </div>
  )
}
