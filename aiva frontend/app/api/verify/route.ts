import { type NextRequest, NextResponse } from "next/server"

// Backend API URL - change this to your Python backend URL
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("document") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Forward the request to our Python backend
    const backendFormData = new FormData()
    backendFormData.append("document", file)

    // Create AbortController for timeout handling
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 30000) // 30 second timeout

    try {
      const response = await fetch(`${BACKEND_URL}/api/verify`, {
        method: "POST",
        body: backendFormData,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: "Backend verification failed" }))
        throw new Error(errorData.detail || "Backend verification failed")
      }

      const result = await response.json()
      return NextResponse.json(result)
    } catch (fetchError: any) {
      clearTimeout(timeoutId)
      
      if (fetchError.name === 'AbortError') {
        throw new Error("Request timeout - backend server is taking too long to respond")
      }
      
      if (fetchError.code === 'UND_ERR_HEADERS_TIMEOUT') {
        throw new Error("Connection timeout - please check if backend server is running")
      }
      
      throw fetchError
    }
  } catch (error) {
    console.error("Verification error:", error)
    
    // Provide more specific error messages
    let errorMessage = "Verification failed"
    if (error instanceof Error) {
      if (error.message.includes("timeout")) {
        errorMessage = "Backend server timeout - please try again"
      } else if (error.message.includes("fetch failed")) {
        errorMessage = "Cannot connect to backend server - please check if it's running"
      } else {
        errorMessage = error.message
      }
    }
    
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 }
    )
  }
}
