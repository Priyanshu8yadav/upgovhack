// Bal-Setu Uttar Pradesh State Command Dashboard Engine
document.addEventListener("DOMContentLoaded", () => {
  let deicChart = null;
  let fourDChart = null;
  let careFunnelChart = null;
  let currentReferralPayload = null;
  let allDistricts = [];

  // 1. Initialize Dashboard Charts
  function initCharts() {
    // Chart 1: The 12-DEIC Bottleneck Prevention
    const ctxDeic = document.getElementById("deicComparisonChart").getContext("2d");
    deicChart = new Chart(ctxDeic, {
      type: "bar",
      data: {
        labels: [
          "Naive 5% False-Pos Model",
          "Bal-Setu DEIC Referrals",
          "Bal-Setu Local AAM/NRC",
          "UP 12 DEIC Annual Ceiling"
        ],
        datasets: [{
          label: "वार्षिक रेफ़रल संख्या (Annual Referrals in Lakhs)",
          data: [48.34, 4.34, 25.29, 0.90],
          backgroundColor: [
            "#EF4444", // Naive Red
            "#3B82F6", // Bal-Setu DEIC Blue
            "#10B981", // Local AAM/NRC Emerald
            "#64748B"  // 12 DEIC Ceiling Slate
          ],
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw} Lakh Referrals / Year`
            }
          }
        },
        scales: {
          y: {
            title: { display: true, text: "लाख रेफ़रल / वर्ष (Lakhs / Year)" },
            beginAtZero: true
          }
        }
      }
    });

    // Chart 2: 4D Health Morbidity
    const ctx4D = document.getElementById("fourDChart").getContext("2d");
    fourDChart = new Chart(ctx4D, {
      type: "doughnut",
      data: {
        labels: [
          "D1: जन्मजात विकार (Defects)",
          "D2: पोषण की कमी (Deficiencies)",
          "D3: बचपन के रोग (Diseases)",
          "D4: विकास में विलंब (Delays)"
        ],
        datasets: [{
          data: [3.2, 38.5, 34.2, 24.1],
          backgroundColor: ["#F43F5E", "#F59E0B", "#10B981", "#6366F1"],
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 12, font: { size: 11 } } }
        }
      }
    });

    // Chart 3: RBSK 2.0 Closed-Loop Care Funnel
    const ctxFunnel = document.getElementById("careFunnelChart").getContext("2d");
    careFunnelChart = new Chart(ctxFunnel, {
      type: "bar",
      data: {
        labels: [
          "Screened",
          "Referred",
          "Facility Reached",
          "Confirmed",
          "Treatment Initiated"
        ],
        datasets: [{
          label: "Care Funnel (% Conversion)",
          data: [100.0, 11.6, 9.8, 9.0, 8.6], // High retention
          backgroundColor: "#0D9488",
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.raw}% of Screened Cohort`
            }
          }
        },
        scales: {
          x: { beginAtZero: true, max: 100 }
        }
      }
    });
  }

  // 2. Fetch District Master Table
  async function loadDistrictTable() {
    try {
      const resp = await fetch("/api/stats/districts");
      const data = await resp.json();
      allDistricts = Object.values(data);
      renderDistrictTable(allDistricts);
    } catch (err) {
      console.warn("Using offline fallback district data:", err);
    }
  }

  function renderDistrictTable(distList) {
    const tbody = document.getElementById("districtTableBody");
    tbody.innerHTML = "";
    distList.forEach(d => {
      const tr = document.createElement("tr");
      const deicBadge = d.has_deic 
        ? `<span class="badge badge-success">✓ 1 Operational</span>`
        : `<span class="badge badge-danger">Pending (0)</span>`;
      
      const stunting = d.stunting_rate ? `${(d.stunting_rate * 100).toFixed(0)}%` : "40%";
      const screened = d.screened ? d.screened.toLocaleString() : "670";
      const naive = d.naive_deic_referrals ? d.naive_deic_referrals.toLocaleString() : "85";
      const balsetu = d.balsetu_deic_referrals ? d.balsetu_deic_referrals.toLocaleString() : "8";
      const local = d.balsetu_local_aam_nrc ? d.balsetu_local_aam_nrc.toLocaleString() : "42";
      const rti = d.balsetu_yellow_rti ? d.balsetu_yellow_rti.toLocaleString() : "28";

      tr.innerHTML = `
        <td><strong>${d.name}</strong> <span class="text-muted">(${d.lgd || ""})</span></td>
        <td>${d.division || "UP"}</td>
        <td>${deicBadge}</td>
        <td>${screened}</td>
        <td><strong>${stunting}</strong></td>
        <td class="text-red"><strong>${naive}</strong></td>
        <td class="text-blue"><strong>${balsetu}</strong></td>
        <td class="text-emerald"><strong>${local}</strong></td>
        <td>${rti}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // District Search Filter
  const districtSearch = document.getElementById("districtSearch");
  if (districtSearch) {
    districtSearch.addEventListener("input", (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = allDistricts.filter(d => 
        d.name.toLowerCase().includes(q) || 
        (d.division && d.division.toLowerCase().includes(q))
      );
      renderDistrictTable(filtered);
    });
  }

  // 3. Archetype Switcher
  const archetypes = {
    green: {
      id: "UP-LKO-NORM-01", age: 24, gender: "M", weight: 12.2, height: 87.0, muac: 14.5, hb: 11.8,
      cleft: false, clubfoot: false, squint: 0.0, cataract: false, dental: false, skin: null, speechDelay: false
    },
    yellow: {
      id: "UP-VNS-RTI-02", age: 36, gender: "F", weight: 12.8, height: 92.0, muac: 13.8, hb: 10.4,
      cleft: false, clubfoot: false, squint: 0.0, cataract: false, dental: false, skin: null, speechDelay: true
    },
    sam: {
      id: "UP-BHR-NRC-03", age: 18, gender: "M", weight: 7.4, height: 78.0, muac: 11.0, hb: 8.2,
      cleft: false, clubfoot: false, squint: 0.0, cataract: false, dental: false, skin: null, speechDelay: false
    },
    squint: {
      id: "UP-JHN-AMB-04", age: 48, gender: "F", weight: 15.0, height: 99.0, muac: 14.2, hb: 11.2,
      cleft: false, clubfoot: false, squint: 1.6, cataract: false, dental: false, skin: null, speechDelay: false
    },
    cataract: {
      id: "UP-GKP-DEIC-05", age: 9, gender: "M", weight: 8.1, height: 71.0, muac: 13.4, hb: 11.0,
      cleft: false, clubfoot: false, squint: 0.0, cataract: true, dental: false, skin: null, speechDelay: false
    }
  };

  document.querySelectorAll(".btn-archetype").forEach(btn => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-type");
      const arch = archetypes[type];
      if (!arch) return;

      document.getElementById("inpChildId").value = arch.id;
      document.getElementById("inpAge").value = arch.age;
      document.getElementById("inpGender").value = arch.gender;
      document.getElementById("inpWeight").value = arch.weight;
      document.getElementById("inpHeight").value = arch.height;
      document.getElementById("inpMuac").value = arch.muac;
      document.getElementById("inpHb").value = arch.hb;

      document.getElementById("chkCleft").checked = arch.cleft;
      document.getElementById("chkClubfoot").checked = arch.clubfoot;
      document.getElementById("chkSquint").checked = arch.squint > 0;
      document.getElementById("chkCataract").checked = arch.cataract;
      document.getElementById("chkDental").checked = arch.dental;
      document.getElementById("chkSkin").checked = arch.skin !== null;
      document.getElementById("chkSpeechDelay").checked = arch.speechDelay;

      // Auto trigger screening
      document.getElementById("btnRunScreening").click();
    });
  });

  // 4. Run Screening Action
  const btnRun = document.getElementById("btnRunScreening");
  btnRun.addEventListener("click", async () => {
    btnRun.disabled = true;
    btnRun.textContent = "⏳ Running LiteRT & Clinical Triage...";

    const reqData = {
      child_id: document.getElementById("inpChildId").value,
      district_name: document.getElementById("inpDistrict").value,
      district_lgd: 157,
      age_months: parseFloat(document.getElementById("inpAge").value),
      gender: document.getElementById("inpGender").value,
      weight_kg: parseFloat(document.getElementById("inpWeight").value),
      height_cm: parseFloat(document.getElementById("inpHeight").value),
      muac_cm: parseFloat(document.getElementById("inpMuac").value) || null,
      hb_g_dl: parseFloat(document.getElementById("inpHb").value) || null,
      has_cleft: document.getElementById("chkCleft").checked,
      has_clubfoot: document.getElementById("chkClubfoot").checked,
      has_ntd: false,
      simulated_esotropia_mm: document.getElementById("chkSquint").checked ? 1.5 : 0.0,
      has_cataract: document.getElementById("chkCataract").checked,
      has_bitot_spot: false,
      skin_condition: document.getElementById("chkSkin").checked ? "Scabies" : null,
      dental_caries: document.getElementById("chkDental").checked,
      otitis_media: false,
      lest_passed: !document.getElementById("chkSpeechDelay").checked,
      tdsc_milestones: null,
      mchat_high: false,
      chd_symptoms: false,
      hearing_flag: false,
      seizure_spells: false,
      screen_time_hours: 1.0
    };

    try {
      const resp = await fetch("/api/screen/child", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(reqData)
      });
      const data = await resp.json();
      updateUIWithScreeningResult(data);
    } catch (err) {
      console.error("Screening error:", err);
      alert("Error contacting screening API. Check console.");
    } finally {
      btnRun.disabled = false;
      btnRun.textContent = "⚡ Run On-Device Triage & ABDM Bundle";
    }
  });

  function updateUIWithScreeningResult(data) {
    const summary = data.triage_summary;
    const qrB64 = data.referral_qr_base64;
    const token = data.referral_token;
    const fhir = data.fhir_bundle;

    currentReferralPayload = token;

    // Update 4D Pills
    const setPill = (id, color, label) => {
      const el = document.getElementById(id);
      el.className = `d-pill pill-${color.toLowerCase()}`;
      el.textContent = `${label}: ${color}`;
    };
    setPill("pillD1", summary.d1_defects_color, "D1 जन्मजात");
    setPill("pillD2", summary.d2_deficiencies_color, "D2 पोषण");
    setPill("pillD3", summary.d3_diseases_color, "D3 रोग");
    setPill("pillD4", summary.d4_developmental_color, "D4 विकास");

    // Overall Banner
    const banner = document.getElementById("overallBanner");
    banner.textContent = `समग्र स्थिति: ${summary.overall_parent_color.toUpperCase()}`;
    banner.style.background = summary.overall_parent_color === "Green" ? "#DCFCE7" : (summary.overall_parent_color === "Yellow" ? "#FEF3C7" : "#FFE4E6");
    banner.style.color = summary.overall_parent_color === "Green" ? "#166534" : (summary.overall_parent_color === "Yellow" ? "#92400E" : "#9F1239");

    // Audio Transcript
    document.getElementById("audioTranscriptText").textContent = summary.plain_hindi_audio_summary;

    // Routing Badge
    const badge = document.getElementById("routingBadge");
    badge.innerHTML = `<strong>क्लिनिकल रूटिंग:</strong> ${summary.clinical_routing_lane} → <em>${summary.destination_facility_type}</em>`;

    // QR Area
    const qrDisplay = document.getElementById("qrDisplayArea");
    const scanAction = document.getElementById("facilityScanAction");
    const ashaToast = document.getElementById("ashaCreditToast");
    ashaToast.style.display = "none";

    if (qrB64) {
      qrDisplay.innerHTML = `
        <img src="${qrB64}" class="qr-image" alt="Referral QR">
        <p style="font-size:11px; margin-top:4px;"><strong>${token.referral_id}</strong><br>HMAC-SHA256: <code>${token.signature}</code></p>
      `;
      scanAction.style.display = "block";
    } else {
      qrDisplay.innerHTML = `<p class="text-muted" style="padding:20px;">ग्रीन परिणाम: रेफरल की आवश्यकता नहीं है।</p>`;
      scanAction.style.display = "none";
    }

    // FHIR JSON Display
    document.getElementById("fhirJsonDisplay").textContent = JSON.stringify(fhir, null, 2);
  }

  // 5. Spoken Audio Playback
  const btnPlayAudio = document.getElementById("btnPlayAudio");
  btnPlayAudio.addEventListener("click", () => {
    const text = document.getElementById("audioTranscriptText").textContent;
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "hi-IN";
      u.rate = 0.9;
      window.speechSynthesis.speak(u);
    } else {
      alert("Web Speech API not supported on this browser.");
    }
  });

  // 6. Simulate Facility Scan
  const btnScan = document.getElementById("btnSimulateScan");
  btnScan.addEventListener("click", async () => {
    if (!currentReferralPayload) return;
    btnScan.disabled = true;
    btnScan.textContent = "Scanning at Hospital Desk...";

    try {
      const resp = await fetch("/api/referral/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_qr_data: JSON.stringify(currentReferralPayload),
          clerk_id: "CLERK-LKO-CHC-01"
        })
      });
      const res = await resp.json();
      if (res.success) {
        const toast = document.getElementById("ashaCreditToast");
        toast.style.display = "block";
        toast.innerHTML = `✓ Arrival Verified! ASHA incentive of <strong>₹${res.record.asha_incentive.amount_inr}</strong> credited via PFMS.`;
      }
    } catch (err) {
      alert("Facility scan failed: " + err);
    } finally {
      btnScan.disabled = false;
      btnScan.textContent = "🏥 Simulate Facility Desk QR Scan";
    }
  });

  // 7. Interactive Policy Simulator Engine
  let simDeics = 12;
  let simFpRate = 0.05;
  let simMode = "balsetu_4tier";

  const sliderDeic = document.getElementById("sliderSimDeic");
  const sliderFp = document.getElementById("sliderSimFp");
  const btnBalsetu = document.getElementById("btnModeBalsetu");
  const btnNaive = document.getElementById("btnModeNaive");

  function updateSimulation() {
    const screenings = 28433333;
    const deicCeiling = simDeics * 7500;
    const deicCeilingLakh = (deicCeiling / 100000).toFixed(2);

    // Naive model
    const naiveFp = Math.round(screenings * simFpRate);
    const naiveTrue = Math.round(screenings * 0.05);
    const naiveTotal = naiveFp + naiveTrue;
    const naiveTotalLakh = (naiveTotal / 100000).toFixed(2);
    const naiveOverload = (naiveTotal / Math.max(1, deicCeiling)).toFixed(1);

    // Bal-Setu model
    const balsetuDeicTrue = Math.round(screenings * 0.0028);
    const balsetuDeicFp = Math.round(screenings * (simFpRate * 0.016));
    const balsetuDeicTotal = balsetuDeicTrue + balsetuDeicFp;
    const balsetuDeicLakh = (balsetuDeicTotal / 100000).toFixed(2);
    const balsetuOverload = (balsetuDeicTotal / Math.max(1, deicCeiling)).toFixed(2);
    const balsetuAamLakh = ((screenings * 0.089) / 100000).toFixed(2);

    const savingsCr = (((naiveFp - balsetuDeicFp) * 450) / 10000000).toFixed(1);

    // Update text readouts
    document.getElementById("txtSimDeicCount").textContent = `${simDeics} / 75 Districts`;
    document.getElementById("txtSimFpRate").textContent = `${(simFpRate * 100).toFixed(1)}%`;
    document.getElementById("kpiSimCapacity").textContent = `${(deicCeiling / 1000).toFixed(0)}k / Yr`;

    const isBalsetu = (simMode === "balsetu_4tier");
    const currentReferrals = isBalsetu ? `${balsetuDeicLakh} Lakh` : `${naiveTotalLakh} Lakh`;
    const currentOverload = isBalsetu ? `${balsetuOverload}x` : `${naiveOverload}x`;

    const elReferrals = document.getElementById("kpiSimReferrals");
    const elOverload = document.getElementById("kpiSimOverload");
    const elBadge = document.getElementById("kpiSimOverloadBadge");
    const elSub = document.getElementById("kpiSimReferralsSub");

    elReferrals.textContent = currentReferrals;
    elOverload.textContent = currentOverload;
    document.getElementById("kpiSimSavings").textContent = `₹${savingsCr} Cr`;

    if (parseFloat(currentOverload) <= 1.0) {
      elBadge.className = "sim-badge-safe";
      elBadge.textContent = "SAFE / OPTIMAL";
      elReferrals.style.color = "#10B981";
      elSub.textContent = "Within clinical intake capacity";
    } else if (parseFloat(currentOverload) <= 2.5) {
      elBadge.className = "sim-badge-safe";
      elBadge.style.background = "#FEF3C7";
      elBadge.style.color = "#B45309";
      elBadge.textContent = "MODERATE STRAIN";
      elReferrals.style.color = "#F59E0B";
      elSub.textContent = "Requires mild shift-scheduling";
    } else {
      elBadge.className = "sim-badge-danger";
      elBadge.textContent = "CRITICAL DDOS OVERLOAD";
      elReferrals.style.color = "#E11D48";
      elSub.textContent = `${currentOverload} clinical ceiling!`;
    }

    // Live update Chart 1
    if (deicChart) {
      deicChart.data.datasets[0].data = [
        parseFloat(naiveTotalLakh),
        parseFloat(balsetuDeicLakh),
        parseFloat(balsetuAamLakh),
        parseFloat(deicCeilingLakh)
      ];
      deicChart.update();
    }
  }

  if (sliderDeic) {
    sliderDeic.addEventListener("input", (e) => {
      simDeics = parseInt(e.target.value);
      updateSimulation();
    });
  }

  if (sliderFp) {
    sliderFp.addEventListener("input", (e) => {
      simFpRate = parseFloat(e.target.value) / 100.0;
      updateSimulation();
    });
  }

  if (btnBalsetu && btnNaive) {
    btnBalsetu.addEventListener("click", () => {
      simMode = "balsetu_4tier";
      btnBalsetu.classList.add("active");
      btnNaive.classList.remove("active");
      updateSimulation();
    });

    btnNaive.addEventListener("click", () => {
      simMode = "naive_binary";
      btnNaive.classList.add("active");
      btnBalsetu.classList.remove("active");
      updateSimulation();
    });
  }

  // 8. Live Closed-Loop Referral Queue & Facility Arrival Scanner
  async function loadLiveReferralQueue() {
    try {
      const res = await fetch("/api/referrals/live");
      const data = await res.json();
      if (data.summary) {
        const setVal = (id, val) => {
          const el = document.getElementById(id);
          if (el) el.innerText = val;
        };
        setVal("liveTotalScreened", data.summary.total_screenings_recorded);
        setVal("liveActiveReferrals", data.summary.active_in_transit_referrals);
        setVal("liveVerifiedArrivals", data.summary.facility_arrivals_verified);
        setVal("liveAamDiversions", data.summary.local_aam_routed_count);
      }
      const tbody = document.getElementById("liveReferralsTableBody");
      if (tbody && data.recent_referrals) {
        if (data.recent_referrals.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #94a3b8; padding: 16px;">No recent referrals recorded yet. Run a screening in Frontline Screener to populate live queue.</td></tr>';
        } else {
          window.latestReferralsList = data.recent_referrals;
          tbody.innerHTML = data.recent_referrals.map((r, idx) => `
            <tr>
              <td><strong>${r.child_id}</strong></td>
              <td><code>${r.token_id}</code></td>
              <td><span class="badge ${r.clinical_lane.includes("DEIC") ? "badge-danger" : "badge-warning"}">${r.clinical_lane}</span></td>
              <td>${r.destination_facility}</td>
              <td>
                <span class="badge ${r.status === "ARRIVED" ? "badge-success" : "badge-warning"}">
                  ${r.status === "ARRIVED" ? "✓ ARRIVED (Verified)" : "⏳ IN TRANSIT (Referred)"}
                </span>
              </td>
              <td>
                ${r.status !== "ARRIVED" ? 
                  `<button class="btn btn-sm btn-primary" onclick="window.triggerLiveArrivalScanByIndex(${idx})">Scan Arrival</button>` : 
                  `<span style="color: #10B981; font-size: 11px; font-weight: 700;">✓ ₹250 PFMS Released</span>`
                }
              </td>
            </tr>
          `).join('');
        }
      }
    } catch (e) {
      console.log("Live referrals note:", e);
    }
  }

  window.triggerLiveArrivalScanByIndex = async function(idx) {
    const r = window.latestReferralsList && window.latestReferralsList[idx];
    if (!r) return;
    const rawToken = r.raw_token_data || r.token_id;
    try {
      const resp = await fetch("/api/referral/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          raw_qr_data: rawToken,
          clerk_id: "CLERK-LKO-CHC-01"
        })
      });
      const res = await resp.json();
      if (res.success) {
        const childIdentifier = r.child_id || res.record?.child_id_hash || res.record?.referral_id || "Child";
        alert("🏥 Facility Desk Scan Success!\n\nChild Arrival Confirmed: " + childIdentifier + "\nASHA Incentive Credited via PFMS: ₹" + (res.record?.asha_incentive?.amount_inr || 250));
        loadLiveReferralQueue();
      } else {
        alert("Scan rejected: " + (res.detail || res.message));
      }
    } catch (err) {
      alert("Scan error: " + err);
    }
  };

  // Poll live referrals every 5 seconds
  setInterval(loadLiveReferralQueue, 5000);

  // Start initialization
  initCharts();
  loadDistrictTable();
  updateSimulation();
  loadLiveReferralQueue();
});
