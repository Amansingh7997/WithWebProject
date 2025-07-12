document.addEventListener("DOMContentLoaded", () => {
  const API_URL = "http://localhost:8000/detect_burnout";

  const feedbackInput = document.getElementById("feedbackInput") as HTMLTextAreaElement;
  const submitBtn = document.getElementById("submitBtn") as HTMLButtonElement;
  const resultArea = document.getElementById("resultArea") as HTMLDivElement;

  submitBtn.addEventListener("click", async () => {
    const text = feedbackInput.value.trim();

    if (!text) {
      resultArea.textContent = "⚠️ Please enter your feedback.";
      resultArea.className = "mt-4 text-center text-red-600";
      return;
    }

    resultArea.textContent = "🧪 Analyzing your feedback...";
    resultArea.className = "mt-4 text-center text-gray-700";

    try {
      const response = await fetch("http://0.0.0.0:8000", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
      });

      const data = await response.json();
      const emoji = data.prediction === "Burnout Detected" ? "🔥" : "✅";

      resultArea.textContent = `${emoji} ${data.prediction} (Score: ${data.score})`;
      resultArea.className = `mt-4 text-center ${data.prediction === "Burnout Detected" ? "text-red-500" : "text-green-600"} font-semibold`;
    } catch (err) {
      console.error(err);
      resultArea.textContent = "❌ Error contacting server.";
      resultArea.className = "mt-4 text-center text-red-600";
    }
  });
});
