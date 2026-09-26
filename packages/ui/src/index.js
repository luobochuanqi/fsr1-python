import "./style.css";

// 前后对比查看器：contain 定位 + 滚轮缩放/拖拽平移 + 分割线拖动。
// 只管图像交互，业务状态（上传、参数、状态条）留给调用方。
export function createCompareViewer(stage) {
  const compare = document.createElement("div");
  compare.className = "sr-compare";
  compare.innerHTML = `
    <div class="sr-zoom">
      <img class="sr-before" alt="原始分辨率" draggable="false">
      <div class="sr-after"><img alt="超分结果" draggable="false"></div>
      <div class="sr-divider"></div>
    </div>
    <span class="sr-tag sr-tag-l"></span>
    <span class="sr-tag sr-tag-r"></span>`;
  stage.appendChild(compare);

  const zoom = compare.querySelector(".sr-zoom");
  const beforeImg = compare.querySelector(".sr-before");
  const afterImg = compare.querySelector(".sr-after img");
  const tagL = compare.querySelector(".sr-tag-l");
  const tagR = compare.querySelector(".sr-tag-r");

  // 框定位：按输出图宽高比在舞台内 contain，滑块与图像严格对齐。
  let fitAR = null;
  function applyFit() {
    if (!fitAR) return;
    const r = stage.getBoundingClientRect();
    const pad = 16;
    const bw = r.width - pad * 2;
    const bh = r.height - pad * 2;
    const cw = Math.min(bw, bh * fitAR);
    compare.style.width = Math.round(cw) + "px";
    compare.style.height = Math.round(cw / fitAR) + "px";
    clampPan();
    applyZoom();
  }
  new ResizeObserver(applyFit).observe(stage);

  // 缩放与平移：#sr-zoom 铺满 compare（overflow hidden），transform-origin: 0 0，
  // transform = translate(t) scale(s)。图坐标 p、compare 内坐标 q 满足 q = p*s + t。
  // 滚轮以光标为锚点：缩放前后光标下的图坐标不变。
  let zs = 1, ztx = 0, zty = 0;
  function applyZoom() {
    zoom.style.transform = `translate(${ztx}px, ${zty}px) scale(${zs})`;
  }
  // 平移范围约束：缩放后图像不滑出边界；比视口小时居中。
  function clampPan() {
    const w = compare.offsetWidth, h = compare.offsetHeight;
    const sw = w * zs, sh = h * zs;
    if (sw > w) ztx = Math.min(0, Math.max(w - sw, ztx));
    else ztx = (w - sw) / 2;
    if (sh > h) zty = Math.min(0, Math.max(h - sh, zty));
    else zty = (h - sh) / 2;
  }
  function resetView() {
    zs = 1; ztx = 0; zty = 0;
    applyZoom();
  }

  compare.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = compare.getBoundingClientRect();
    const qx = e.clientX - rect.left, qy = e.clientY - rect.top;
    const f = e.deltaY > 0 ? 1 / 1.12 : 1.12;
    const ns = Math.min(Math.max(zs * f, 1), 24);
    if (ns === zs) return;
    // 锚点图坐标（缩放前），要求缩放后仍对准光标。
    const ax = (qx - ztx) / zs, ay = (qy - zty) / zs;
    zs = ns;
    ztx = qx - ax * zs;
    zty = qy - ay * zs;
    clampPan();
    applyZoom();
  }, { passive: false });

  // 缩放>1 拖拽平移；否则拖分割线。统一在此分发，避免双 pointerdown 冲突。
  let pane = false, plx = 0, ply = 0, raf = 0;
  compare.addEventListener("pointerdown", (e) => {
    pane = zs > 1;
    if (pane) {
      e.preventDefault();
      compare.setPointerCapture(e.pointerId);
      plx = e.clientX; ply = e.clientY;
    } else {
      setX(e.clientX);
    }
  });
  compare.addEventListener("pointermove", (e) => {
    if (e.buttons === 0) return;
    if (pane) {
      const dx = e.clientX - plx, dy = e.clientY - ply;
      plx = e.clientX; ply = e.clientY;
      ztx += dx; zty += dy;
      clampPan();
      applyZoom();
    } else {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => setX(e.clientX));
    }
  });
  compare.addEventListener("pointerup", () => { pane = false; });
  compare.addEventListener("pointercancel", () => { pane = false; });

  // 双击复位。
  compare.addEventListener("dblclick", resetView);

  function setX(clientX) {
    const r = compare.getBoundingClientRect();
    const x = Math.min(Math.max((clientX - r.left) / r.width, 0), 1) * 100;
    compare.style.setProperty("--x", x + "%");
  }

  return {
    // 切换图对：复位视图，按 after 尺寸适配框。两图都加载完才 resolve。
    setPair(beforeUrl, afterUrl) {
      resetView();
      return new Promise((resolve, reject) => {
        afterImg.onload = () => {
          fitAR = afterImg.naturalWidth / afterImg.naturalHeight;
          applyFit();
          beforeImg.onload = resolve;
          beforeImg.onerror = () => reject(new Error("before 图片加载失败"));
          beforeImg.src = beforeUrl;
        };
        afterImg.onerror = () => reject(new Error("after 图片加载失败"));
        afterImg.src = afterUrl;
      });
    },
    setLabels(left, right) {
      tagL.textContent = left;
      tagR.textContent = right;
    },
  };
}
