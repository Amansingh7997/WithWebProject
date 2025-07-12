const API_URL = "http://127.0.0.1:8000/predict"; // Replace with actual URL from ngrok

document.getElementById("submitBtn").addEventListener("click", async () => {
  const feedbackInput = document.getElementById("feedbackInput").value;
  const resultArea = document.getElementById("resultArea");

  resultArea.textContent = "Analyzing...";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text: feedbackInput })
    });

    if (!response.ok) throw new Error("Server error");

    const data = await response.json();
    resultArea.textContent = `🔥 Burnout Level: ${data.burnout_level}`;
  } catch (error) {
    console.error("Error:", error);
    resultArea.textContent = "⚠️ Failed to get prediction. Try again.";
  }
});
