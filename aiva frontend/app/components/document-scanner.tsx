"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Upload, FileText, Loader2, Shield } from "lucide-react"

interface DocumentScannerProps {
  onDocumentUpload: (file: File) => void
  isProcessing: boolean
  currentStep: string
}

export default function DocumentScanner({ onDocumentUpload, isProcessing, currentStep }: DocumentScannerProps) {
  const [dragActive, setDragActive] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (file: File) => {
    if (file.type.startsWith("image/")) {
      setSelectedFile(file)
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0])
    }
  }

  const startVerification = () => {
    if (selectedFile) {
      onDocumentUpload(selectedFile)
    }
  }

  const getStepMessage = () => {
    switch (currentStep) {
      case "scanning":
        return "Processing document with computer vision..."
      case "ai-analysis":
        return "Analyzing document authenticity with AI..."
      case "blockchain":
        return "Recording verification on blockchain..."
      case "complete":
        return "Verification complete!"
      default:
        return "Upload a document to begin verification"
    }
  }

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <FileText className="h-5 w-5" />
          Document Scanner
        </CardTitle>
        <CardDescription>{getStepMessage()}</CardDescription>
      </CardHeader>
      <CardContent>
        {!selectedFile ? (
          <div
            className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
              dragActive ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-gray-400"
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">Upload Document</h3>
            <p className="text-gray-500 mb-4">Drag and drop your document here, or click to browse</p>
            <p className="text-sm text-gray-400 mb-4">Supports: Aadhaar, PAN, Passport, Driving License, etc.</p>
            <Button onClick={() => fileInputRef.current?.click()} variant="outline">
              Choose File
            </Button>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} className="hidden" />
          </div>
        ) : (
          <div className="space-y-4">
            {/* Document Preview */}
            <div className="relative">
              <img
                src={previewUrl || ""}
                alt="Document preview"
                className="w-full h-64 object-contain bg-gray-50 rounded-lg border"
              />
              {isProcessing && (
                <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center rounded-lg">
                  <div className="text-white text-center">
                    <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
                    <p className="text-sm">Processing...</p>
                  </div>
                </div>
              )}
            </div>

            {/* File Info */}
            <div className="bg-gray-50 p-3 rounded-lg">
              <p className="text-sm font-medium">{selectedFile.name}</p>
              <p className="text-xs text-gray-500">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-2">
              <Button onClick={startVerification} disabled={isProcessing} className="flex-1">
                {isProcessing ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Verifying...
                  </>
                ) : (
                  <>
                    <Shield className="h-4 w-4 mr-2" />
                    Start Verification
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  setSelectedFile(null)
                  setPreviewUrl(null)
                }}
                disabled={isProcessing}
              >
                Clear
              </Button>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
