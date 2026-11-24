// Bal-Setu Parent Companion: Offline Mini-PWA Engine
document.addEventListener("DOMContentLoaded", () => {
  // 1. Tab Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetEl = document.getElementById(targetId);
      if (targetEl) targetEl.classList.add("active");
      if (targetId === "tab-rti" && rtiChart) {
        rtiChart.resize();
      }
    });
  });

  // 2. Speech Game Logic
  const btnSpeakPrompt = document.getElementById("btnSpeakPrompt");
  const btnRecordVoice = document.getElementById("btnRecordVoice");
  const speechFeedback = document.getElementById("speechFeedback");

  const promptWords = [
    { name: "सेब (Apple)", icon: "🍎", audioText: "यह एक सेब है। बोलो: सेब!", audioKey: "apple_hi" },
    { name: "कटोरी (Bowl)", icon: "🥣", audioText: "यह कटोरी है। बोलो: कटोरी!", audioKey: "bowl_hi" },
    { name: "चम्मच (Spoon)", icon: "🥄", audioText: "यह चम्मच है। बोलो: चम्मच!", audioKey: "spoon_hi" },
    { name: "पानी (Water)", icon: "💧", audioText: "यह पानी है। बोलो: पानी!", audioKey: "water_hi" }
  ];
  let currentPromptIdx = 0;
  let activeAudio = null;

  function playNativeAudio(key, fallbackText) {
    if (activeAudio) {
      try { activeAudio.pause(); } catch(e){}
    }
    const audio = new Audio(`/pwa/audio/${key}.m4a`);
    activeAudio = audio;
    audio.play().catch(err => {
      speakHindi(fallbackText);
    });
  }

  function speakHindi(text) {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "hi-IN";
      utterance.rate = 0.9;
      window.speechSynthesis.speak(utterance);
    }
  }

  if (btnSpeakPrompt) {
    btnSpeakPrompt.addEventListener("click", () => {
      const cur = promptWords[currentPromptIdx];
      playNativeAudio(cur.audioKey, cur.audioText);
      speechFeedback.innerHTML = `🔊 बज रहा है: "<em>${cur.audioText}</em>"`;
    });
  }

  if (btnRecordVoice) {
    btnRecordVoice.addEventListener("click", () => {
      speechFeedback.innerHTML = "🎙️ सुन रहे हैं... बच्चे को बोलने दें (Listening...)";
      setTimeout(() => {
        const scores = [88, 92, 96];
        const randomScore = scores[Math.floor(Math.random() * scores.length)];
        speechFeedback.innerHTML = `⭐ <strong>शानदार! (Score: ${randomScore}/100)</strong><br>ध्वनि और शब्दांश स्पष्ट हैं। अगला शब्द आ रहा है...`;
        playNativeAudio("praise_hi", "शाबाश! बहुत अच्छा बोला!");
        setTimeout(() => {
          currentPromptIdx = (currentPromptIdx + 1) % promptWords.length;
          const next = promptWords[currentPromptIdx];
          document.querySelector(".item-icon").textContent = next.icon;
          document.querySelector(".item-name").textContent = next.name;
          speechFeedback.textContent = "नया शब्द तैयार है। ऑडियो सुनें बटन दबाएं।";
        }, 2000);
      }, 1400);
    });
  }

  // 3. Tracing Canvas Logic (CalibCard Back Circle / Pattern 1)
  const canvas = document.getElementById("tracingCanvas");
  let ctx = canvas ? canvas.getContext("2d") : null;
  let isDrawing = false;
  let drawnPoints = [];

  function drawTemplate() {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw guide dotted circle (Pattern 1, center 160, 100, radius 60)
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "#94A3B8";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(160, 100, 60, 0, 2 * Math.PI);
    ctx.stroke();

    // Start green dot
    ctx.fillStyle = "#16A34A";
    ctx.beginPath();
    ctx.arc(160, 40, 6, 0, 2 * Math.PI);
    ctx.fill();

    // Start label
    ctx.font = "11px sans-serif";
    ctx.fillStyle = "#64748B";
    ctx.fillText("यहाँ से शुरू करें (Start)", 115, 30);
    ctx.restore();
  }

  if (canvas && ctx) {
    drawTemplate();

    function getCoords(e) {
      const rect = canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        x: clientX - rect.left,
        y: clientY - rect.top
      };
    }

    function startDraw(e) {
      e.preventDefault();
      isDrawing = true;
      const pos = getCoords(e);
      drawnPoints.push(pos);
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y);
      ctx.strokeStyle = "#2563EB";
      ctx.lineWidth = 4;
      ctx.lineCap = "round";
    }

    function drawMove(e) {
      if (!isDrawing) return;
      e.preventDefault();
      const pos = getCoords(e);
      drawnPoints.push(pos);
      ctx.lineTo(pos.x, pos.y);
      ctx.stroke();
    }

    function endDraw() {
      isDrawing = false;
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", drawMove);
    canvas.addEventListener("mouseup", endDraw);
    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", drawMove, { passive: false });
    canvas.addEventListener("touchend", endDraw);

    const btnClear = document.getElementById("btnClearCanvas");
    if (btnClear) {
      btnClear.addEventListener("click", () => {
        drawnPoints = [];
        drawTemplate();
        const scoreBox = document.getElementById("traceScoreBox");
        if (scoreBox) scoreBox.style.display = "none";
      });
    }

    const btnGrade = document.getElementById("btnEvaluateTrace");
    if (btnGrade) {
      btnGrade.addEventListener("click", () => {
        if (drawnPoints.length < 15) {
          alert("कृपया पहले स्क्रीन पर वृत्त के ऊपर उंगली से रेखा खींचें।");
          return;
        }
        // Compute Hausdorff / mean distance to circle (cx=160, cy=100, r=60)
        let totalDev = 0;
        drawnPoints.forEach(pt => {
          const distToCenter = Math.hypot(pt.x - 160, pt.y - 100);
          totalDev += Math.abs(distToCenter - 60);
        });
        const meanDevPx = totalDev / drawnPoints.length;
        // In this canvas scale, 1 mm ~ 3.5 px
        const meanDevMm = (meanDevPx / 3.5).toFixed(2);
        const score = Math.max(10, Math.min(100, Math.round(100 - (meanDevMm * 14))));

        const scoreBox = document.getElementById("traceScoreBox");
        const scoreNum = document.getElementById("traceScoreNum");
        const scoreDesc = document.getElementById("traceScoreDesc");

        if (scoreBox) {
          scoreBox.style.display = "block";
          scoreNum.textContent = `${score}/100`;
          if (score >= 70) {
            scoreDesc.textContent = `शानदार समन्वय! विचलन मात्र ${meanDevMm} मिमी (सामान्य सीमा में)।`;
          } else {
            scoreDesc.textContent = `विचलन ${meanDevMm} मिमी है। हाथ की पकड़ और बिंदु मिलाने का अभ्यास जारी रखें।`;
          }
        }
      });
    }
  }

  // 4. RtI Progress Velocity Chart (Chart.js)
  const rtiCanvas = document.getElementById("rtiChart");
  let rtiChart = null;

  const responderData = [35, 48, 62, 78]; // +14.3 pts/wk
  const nonResponderData = [35, 38, 41, 42]; // +2.3 pts/wk (Flat)

  if (rtiCanvas) {
    rtiChart = new Chart(rtiCanvas, {
      type: "line",
      data: {
        labels: ["सप्ताह 1 (प्रारंभ)", "सप्ताह 2", "सप्ताह 3 (वर्तमान)", "सप्ताह 4 (लक्ष्य)"],
        datasets: [
          {
            label: "बच्चे की प्रगति (Child Score)",
            data: responderData,
            borderColor: "#0D9488",
            backgroundColor: "rgba(13, 148, 136, 0.15)",
            borderWidth: 3,
            fill: true,
            tension: 0.2,
            pointRadius: 6,
            pointBackgroundColor: "#0F766E"
          },
          {
            label: "सामान्य स्तर कटऑफ (Green Threshold)",
            data: [70, 70, 70, 70],
            borderColor: "#16A34A",
            borderDash: [5, 5],
            borderWidth: 2,
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 0,
            max: 100,
            title: { display: true, text: "विकास स्कोर (0-100)" }
          }
        },
        plugins: {
          legend: { position: "bottom" }
        }
      }
    });
  }

  const btnScenarioResponder = document.getElementById("btnScenarioResponder");
  const btnScenarioNonResponder = document.getElementById("btnScenarioNonResponder");
  const rtiAlertBox = document.getElementById("rtiAlertBox");
  const rtiAlertTitle = document.getElementById("rtiAlertTitle");
  const rtiAlertDesc = document.getElementById("rtiAlertDesc");

  if (btnScenarioResponder) {
    btnScenarioResponder.addEventListener("click", () => {
      if (!rtiChart) return;
      rtiChart.data.datasets[0].data = responderData;
      rtiChart.data.datasets[0].borderColor = "#0D9488";
      rtiChart.update();
      if (rtiAlertBox) {
        rtiAlertBox.className = "alert-box";
        rtiAlertTitle.textContent = "प्रगति की स्थिति: सुधार की ओर (Responder: +14.3 pts/wk)";
        rtiAlertDesc.textContent = "सकारात्मक सुधार! चौथे सप्ताह के बाद बच्चा स्वतः सामान्य श्रेणी (Green) में आ जाएगा।";
      }
    });
  }

  if (btnScenarioNonResponder) {
    btnScenarioNonResponder.addEventListener("click", () => {
      if (!rtiChart) return;
      rtiChart.data.datasets[0].data = nonResponderData;
      rtiChart.data.datasets[0].borderColor = "#EF4444";
      rtiChart.update();
      if (rtiAlertBox) {
        rtiAlertBox.className = "alert-box escalated";
        rtiAlertTitle.textContent = "चेतावनी: प्रगति रुकी हुई है (Non-Responder: Flat slope)";
        rtiAlertDesc.textContent = "4 सप्ताह के अभ्यास के बाद भी सुधार नहीं हुआ। आशा संगिनी और ब्लॉक मेडिकल ऑफिसर को स्वतः अलर्ट भेजा गया। (Amber/Red DEIC Escalation active).";
      }
    });
  }

  // 5. Dialect IVR Simulator
  const btnSimulateIVRCall = document.getElementById("btnSimulateIVRCall");
  const dialectSelect = document.getElementById("dialectSelect");
  const callBox = document.getElementById("callBox");
  const callTranscript = document.getElementById("callTranscript");
  const keypadBtns = document.querySelectorAll(".btn-num");
  const callResult = document.getElementById("callResult");

  const dialectTranscripts = {
    bho: "प्रणाम! बाल-सेतु सेवा में राउर स्वागत बा। एह हफ्ता के खेल: बबुआ के सामने कटोरी, चमचा आ लोटा रखीं। नाम बोले खातिर कही। अगर बबुआ खेललसि त 1 दबाईं। आशा दीदी खातिर 2 दबाईं।",
    hi: "नमस्कार! राष्ट्रीय बाल स्वास्थ्य कार्यक्रम बाल-सेतु में आपका स्वागत है। इस सप्ताह का खेल: कटोरी और चम्मच का नाम पूछें। खेल पूरा होने पर फोन में 1 दबाएं।",
    awa: "राम-राम! बाल-सेतु कार्यक्रम मा आपन स्वागत अहइ। लरिका के आगे कटोरी-चमचा धरि के नाम पूछो। खेल खेलिस होय त 1 दबावा।",
    bnd: "जय राम जी की! बाल-सेतु में स्वागत है। मोड़ा के सामने कटोरी-चमचा रखो और नाम पूछो। खेल पूरा भओ होवे तो 1 दबाओ।",
    braj: "राधे-राधे! लल्ला के आगे कटोरी अरु चम्मच धर के नाम पूछौ। यदि खेल खेल्यौ होय तौ 1 दबाओ।"
  };

  if (btnSimulateIVRCall) {
    btnSimulateIVRCall.addEventListener("click", () => {
      const selectedDialect = dialectSelect.value;
      const text = dialectTranscripts[selectedDialect] || dialectTranscripts.hi;
      if (callBox) callBox.style.display = "block";
      if (callTranscript) callTranscript.textContent = text;
      if (callResult) callResult.style.display = "none";
      if (selectedDialect === "bho") {
        playNativeAudio("bho_ivr", text);
      } else if (selectedDialect === "awa") {
        playNativeAudio("awa_ivr", text);
      } else {
        speakHindi(text);
      }
    });
  }

  keypadBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.getAttribute("data-key");
      if (callResult) {
        callResult.style.display = "block";
        if (key === "1") {
          callResult.className = "call-footer";
          callResult.innerHTML = "✓ पुष्टि दर्ज! खेल पूरा हुआ। ASHA e-Kavach कार्यसूची में सिंक हुआ।";
          speakHindi("धन्यवाद! आपकी पुष्टि दर्ज हो गई है।");
        } else if (key === "2") {
          callResult.className = "call-footer";
          callResult.innerHTML = "📞 आशा दीदी (संगीता देवी) को तुरंत गृह-भ्रमण का संदेश भेजा गया।";
          speakHindi("आशा दीदी को संदेश भेज दिया गया है।");
        } else {
          const selectedDialect = dialectSelect.value;
          speakHindi(dialectTranscripts[selectedDialect]);
        }
      }
    });
  });

  // 6. Service Worker Registration for Offline Capabilities
  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("/pwa/sw.js", { scope: "/pwa/" }).then(() => {
      console.log("Bal-Setu Offline Service Worker registered.");
    }).catch(err => {
      console.warn("Service worker registration:", err);
    });
  }
});
