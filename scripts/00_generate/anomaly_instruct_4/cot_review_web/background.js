"use strict";
(() => {
  const canvas = document.getElementById("ambient-background");
  const button = document.getElementById("toggle-background");
  const context = canvas?.getContext("2d");
  if (!context || !button) return;

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  const preferenceKey = "traceLensAmbientMotion";
  let wanted = true;
  try { wanted = localStorage.getItem(preferenceKey) !== "off"; } catch {}
  let width = 0, height = 0, points = [], frame = 0, lastDraw = 0;

  function paint(step = 0) {
    context.clearRect(0, 0, width, height);
    for (const point of points) {
      point.x += point.vx * step * 3;
      point.y += point.vy * step * 3;
      if (point.x < -15) point.x = width + 15;
      if (point.x > width + 15) point.x = -15;
      if (point.y < -15) point.y = height + 15;
      if (point.y > height + 15) point.y = -15;
    }
    for (let i = 0; i < points.length; i++) {
      const a = points[i];
      let linked = 0;
      for (let j = i + 1; j < points.length && linked < 3; j++) {
        const b = points[j], distance = Math.hypot(a.x - b.x, a.y - b.y);
        if (distance > 150) continue;
        linked++;
        context.strokeStyle = `rgba(72,132,217,${.22 + .28 * (1 - distance / 150)})`;
        context.lineWidth = 2;
        context.beginPath(); context.moveTo(a.x, a.y); context.lineTo(b.x, b.y); context.stroke();
      }
      context.fillStyle = "rgba(84,143,225,.72)";
      context.beginPath(); context.arc(a.x, a.y, a.radius, 0, Math.PI * 2); context.fill();
    }
  }

  function animate(time) {
    if (!frame) return;
    if (time - lastDraw >= 1000 / 30) {
      paint(lastDraw ? Math.min(2, (time - lastDraw) / 33.3) : 0);
      lastDraw = time;
    }
    frame = requestAnimationFrame(animate);
  }

  function sync() {
    const enabled = wanted && !reduced.matches;
    const running = enabled && !document.hidden && !document.body.classList.contains("modal-open");
    button.textContent = `背景动画：${enabled ? "开启" : "关闭"}`;
    button.setAttribute("aria-pressed", String(enabled));
    button.disabled = reduced.matches;
    button.title = reduced.matches ? "系统已启用减少动态效果" : "点击切换背景动画，设置仅保存在此浏览器";
    canvas.dataset.motion = reduced.matches ? "reduced" : running ? "running" : "paused";
    if (frame) cancelAnimationFrame(frame);
    frame = 0; lastDraw = 0;
    if (running) frame = requestAnimationFrame(animate);
    else paint();
  }

  function resize() {
    width = window.innerWidth; height = window.innerHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    const count = Math.min(60, Math.max(22, Math.floor(width * height / 24000)));
    points = Array.from({length: count}, () => ({
      x: Math.random() * width, y: Math.random() * height,
      vx: (Math.random() - .5) * .32, vy: (Math.random() - .5) * .28,
      radius: 2.2 + Math.random() * 1.8
    }));
    canvas.dataset.points = String(count);
    paint(); sync();
  }

  button.addEventListener("click", () => {
    wanted = !wanted;
    try { localStorage.setItem(preferenceKey, wanted ? "on" : "off"); } catch {}
    sync();
  });
  window.addEventListener("resize", resize, {passive: true});
  document.addEventListener("visibilitychange", sync);
  reduced.addEventListener("change", sync);
  new MutationObserver(sync).observe(document.body, {attributes: true, attributeFilter: ["class"]});
  resize();
})();
