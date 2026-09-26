import { createCompareViewer } from "@sr/ui";
import "./style.css";

const $ = (id) => document.getElementById(id);
const statusEl = $("status"), statusText = $("statusText");
const viewer = createCompareViewer($("stage"));
viewer.setLabels("BILINEAR", "FSR1");

// 当前展示状态：mode = "sample"（静态图对）| "upload"（API 实时计算）。
let busy = false;
let mode = "sample", curSample = "genshin";
let bytes = null, name = "";

const SAMPLES = [
  { id: "genshin", label: "Genshin" },
  { id: "tombraider", label: "Tomb Raider" },
];

function setStatus(text, live = false, error = false) {
  statusText.textContent = text;
  statusText.className = live ? "live" : "";
  statusEl.classList.toggle("error", error);
}

// 加载内置示例的图对；origin 一律为 before，2x（4K）超分图为 after。
async function showSample(sampleId) {
  mode = "sample"; curSample = sampleId;
  const s = SAMPLES.find((x) => x.id === sampleId);
  try {
    await viewer.setPair(`/sample/${sampleId}/origin.jpg`, `/sample/${sampleId}/2x.jpg`);
    setStatus(`${s.label} · 1920x1080 → 3840x2160 · FSR1 2x`);
  } catch (e) {
    setStatus(`图片加载失败 ${e.message}`, false, true);
  }
}

// 上传路径：按 2x POST，返回 4K 超分图。
async function runUpload() {
  const scale = "2";
  if (!bytes || busy) return;
  busy = true;
  setStatus(`超分计算中 ${scale}x · 约 1-2 秒`, true);
  try {
    const res = await fetch(`/api/upscale?scale=${scale}&sharpness=${$("sharp").value}`, {
      method: "POST", body: bytes, headers: { "Content-Type": "application/octet-stream" },
    });
    if (!res.ok) throw new Error(await res.text());
    const url = URL.createObjectURL(await res.blob());
    const dim = await new Promise((ok) => {
      const probe = new Image();
      probe.onload = () => ok(`${probe.naturalWidth}x${probe.naturalHeight}`);
      probe.src = url;
    });
    // before 用原图字节，浏览器缩放到输出尺寸，与 after 逐像素对齐。
    await viewer.setPair(URL.createObjectURL(new Blob([bytes])), url);
    setStatus(`${name} → ${dim} · FSR1 ${scale}x · 锐化 ${$("sharp").value}`);
  } catch (e) {
    setStatus(`失败 ${e.message}`, false, true);
  } finally {
    busy = false;
  }
}

// 画廊按钮。
const gallery = $("gallery"), galleryToggle = $("galleryToggle");
galleryToggle.addEventListener("click", () => gallery.classList.toggle("open"));
for (const s of SAMPLES) {
  const b = document.createElement("button");
  b.className = "thumb";
  b.innerHTML = `<img src="/sample/${s.id}/origin.jpg" alt="${s.label}"><span>${s.label}</span>`;
  b.addEventListener("click", () => {
    gallery.classList.remove("open");
    showSample(s.id);
    markThumb();
  });
  gallery.appendChild(b);
}
function markThumb() {
  gallery.querySelectorAll(".thumb").forEach((b, i) =>
    b.classList.toggle("on", SAMPLES[i].id === curSample));
}
markThumb();

// 锐化滑杆：仅上传模式生效（示例是固定 0.2 的预跑结果）。
$("sharp").addEventListener("input", (e) => {
  $("sharpV").textContent = parseFloat(e.target.value).toFixed(2);
});
$("sharp").addEventListener("change", () => { if (mode === "upload") runUpload(); });

// 上传：任何上传进入 upload 模式；超限（>512px 边）会被 API 拒绝并提示。
$("uploadBtn").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", () => {
  const f = $("fileInput").files[0];
  if (!f) return;
  mode = "upload"; bytes = null; name = f.name;
  setStatus("读取文件中", true);
  const reader = new FileReader();
  reader.onload = () => { bytes = reader.result; runUpload(); };
  reader.readAsArrayBuffer(f);
});

showSample("genshin");
