import { NextResponse } from "next/server"

// Backend API URL - change this to your Python backend URL
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000"

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/api/stats`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error("Failed to fetch system statistics")
    }

    const stats = await response.json()
    return NextResponse.json(stats)
  } catch (error) {
    console.error("Stats fetch error:", error)
    return NextResponse.json(
      { error: "Failed to fetch system statistics" },
      { status: 500 }
    )
  }
} 