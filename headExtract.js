/*
 * headExtract.js - browser port of the Python head-extraction backend.
 * Exposes window.HeadExtract.run(file, keepHair, onStage) -> same object the old /extract "done" event returned:
 *   { png, landmarks, checks, info, passed }
 *
 * Replaces:  MediaPipe (py)      -> @mediapipe/tasks-vision FaceLandmarker (same 478 landmark indices)
 *            BiRefNet (torch)    -> transformers.js + RMBG-1.4 (ONNX, WebGPU/WASM)
 *            closed-form matting -> guided-filter matting + blur-fusion colour decontamination
 *            OpenCV / numpy      -> canvas + typed arrays
 */
import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import {
  AutoModel,
  AutoProcessor,
  RawImage,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

env.allowLocalModels = false;

/* ------------------------------------------------------------------ config */
const OUT_MAXSIDE = 1000,
  IMG_MAX = 2400;
const C = {
  JAW_GROW_X: 0.12,
  JAW_GROW_Y: 0.04,
  JAW_DROP: 0.1,
  SIDE_DROP: 0.1,
  SIDE_DROP_HAIR: 0.4,
  EAR_OUT: 0.22,
  EAR_YAW_BOOST: 0.5,
  EAR_LOBE: 0.06,
  FEATHER: 0.015,
};
const CROP_SIDE = 1.25,
  CROP_ABOVE = 1.0,
  CROP_BELOW_CHIN = 0.25;
const IDX = {
  eyeL: [
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
  ],
  eyeR: [
    362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384,
    398,
  ],
  browL: [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  browR: [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
  lipsOuter: [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0,
    37, 39, 40, 185,
  ],
  lipsInner: [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13,
    82, 81, 80, 191,
  ],
  oval: [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
  ],
};
const JAW = [
  234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365,
  397, 288, 361, 323, 454,
];
const NOSE = 1,
  NOSE_BOTTOM = 2,
  CHIN = 152,
  EAR_L = 234,
  EAR_R = 454,
  BRIDGE = 6,
  CHEEK_L = 50,
  CHEEK_R = 280,
  FOREHEAD = 151;

const clamp = (v, a = 0, b = 1) => Math.max(a, Math.min(b, v));
const tick = () => new Promise((r) => setTimeout(r, 0)); // let the UI repaint between stages
const mk = (w, h) => {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
};

/* ------------------------------------------------------------------ lazy models */
let _lmk, _seg;
async function getLandmarker() {
  if (_lmk) return _lmk;
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm",
  );
  const modelAssetPath =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
  for (const delegate of ["GPU", "CPU"]) {
    try {
      _lmk = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath, delegate },
        runningMode: "IMAGE",
        numFaces: 1,
      });
      return _lmk;
    } catch (e) {
      if (delegate === "CPU") throw e;
    }
  }
}
async function getSeg() {
  if (_seg) return _seg;
  const id = "briaai/RMBG-1.4";
  const model = await AutoModel.from_pretrained(id, {
    config: { model_type: "custom" },
  });
  const proc = await AutoProcessor.from_pretrained(id, {
    config: {
      do_normalize: true,
      do_pad: false,
      do_rescale: true,
      do_resize: true,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [1, 1, 1],
      feature_extractor_type: "ImageFeatureExtractor",
      resample: 2,
      rescale_factor: 1 / 255,
      size: { width: 1024, height: 1024 },
    },
  });
  return (_seg = { model, proc });
}

/* ------------------------------------------------------------------ typed-array image ops */
function box(a, w, h, r) {
  // separable box blur, edge replicate
  const t = new Float32Array(w * h),
    o = new Float32Array(w * h),
    n = 2 * r + 1;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    let s = 0;
    for (let x = -r; x <= r; x++) s += a[row + Math.min(w - 1, Math.max(0, x))];
    for (let x = 0; x < w; x++) {
      t[row + x] = s / n;
      s += a[row + Math.min(w - 1, x + r + 1)] - a[row + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let s = 0;
    for (let y = -r; y <= r; y++)
      s += t[Math.min(h - 1, Math.max(0, y)) * w + x];
    for (let y = 0; y < h; y++) {
      o[y * w + x] = s / n;
      s +=
        t[Math.min(h - 1, y + r + 1) * w + x] - t[Math.max(0, y - r) * w + x];
    }
  }
  return o;
}
function gauss(a, w, h, s) {
  // 3 box passes ~ gaussian
  const r = Math.max(1, Math.round((Math.sqrt(4 * s * s + 1) - 1) / 2));
  let o = a;
  for (let i = 0; i < 3; i++) o = box(o, w, h, r);
  return o;
}
function dil(a, w, h, r) {
  // a: 0/1 (or >0.5) -> 0/1 float
  const b = new Float32Array(w * h);
  for (let i = 0; i < b.length; i++) b[i] = a[i] > 0.5 ? 1 : 0;
  const s = box(b, w, h, r),
    thr = 0.5 / ((2 * r + 1) * (2 * r + 1)),
    o = new Float32Array(w * h);
  for (let i = 0; i < o.length; i++) o[i] = s[i] > thr ? 1 : 0;
  return o;
}
function ero(a, w, h, r) {
  const inv = new Float32Array(w * h);
  for (let i = 0; i < inv.length; i++) inv[i] = a[i] > 0.5 ? 0 : 1;
  const d = dil(inv, w, h, r);
  for (let i = 0; i < d.length; i++) d[i] = 1 - d[i];
  return d;
}
function comps(bin, w, h) {
  // 8-connected labelling
  const lab = new Int32Array(w * h),
    st = new Int32Array(w * h),
    sizes = [0];
  let n = 0;
  for (let i = 0; i < w * h; i++) {
    if (!bin[i] || lab[i]) continue;
    n++;
    let sp = 0,
      c = 0;
    st[sp++] = i;
    lab[i] = n;
    while (sp) {
      const p = st[--sp];
      c++;
      const x = p % w,
        y = (p - x) / w;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const q = yy * w + xx;
          if (bin[q] && !lab[q]) {
            lab[q] = n;
            st[sp++] = q;
          }
        }
      }
    }
    sizes.push(c);
  }
  return { lab, sizes, n };
}
function keepFace(a, w, h, seed, radius) {
  // blob containing the nose, slightly dilated + softened
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) bin[i] = a[i] > 0.5 ? 1 : 0;
  const { lab, sizes, n } = comps(bin, w, h);
  if (!n) return new Float32Array(w * h);
  const sx = clamp(Math.round(seed[0]), 0, w - 1),
    sy = clamp(Math.round(seed[1]), 0, h - 1);
  let l = lab[sy * w + sx];
  if (!l) {
    let m = 0;
    for (let i = 1; i <= n; i++)
      if (sizes[i] > m) {
        m = sizes[i];
        l = i;
      }
  }
  const main = new Float32Array(w * h);
  for (let i = 0; i < main.length; i++) main[i] = lab[i] === l ? 1 : 0;
  const k = Math.max(3, Math.trunc(radius) | 1);
  return gauss(dil(main, w, h, k >> 1), w, h, Math.max(1, k / 4));
}
function polyMask(w, h, polys) {
  const c = mk(w, h),
    g = c.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, w, h);
  g.fillStyle = "#fff";
  for (const p of polys) {
    g.beginPath();
    p.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath();
    g.fill();
  }
  const d = g.getImageData(0, 0, w, h).data,
    o = new Float32Array(w * h);
  for (let i = 0; i < o.length; i++) o[i] = d[4 * i] / 255;
  return o;
}
const mean2 = (pts) => [
  pts.reduce((s, p) => s + p[0], 0) / pts.length,
  pts.reduce((s, p) => s + p[1], 0) / pts.length,
];
function faceCore(w, h, lp, scale) {
  const ov = IDX.oval.map((i) => lp[i]),
    c = mean2(ov);
  return polyMask(w, h, [
    ov.map((p) => [c[0] + (p[0] - c[0]) * scale, c[1] + (p[1] - c[1]) * scale]),
  ]);
}

/* ------------------------------------------------------------------ image / face helpers */
async function loadImage(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); // EXIF-corrected
  const m = Math.max(bmp.width, bmp.height),
    s = m > IMG_MAX ? IMG_MAX / m : 1;
  const c = mk(Math.round(bmp.width * s), Math.round(bmp.height * s)),
    g = c.getContext("2d");
  g.imageSmoothingQuality = "high";
  g.drawImage(bmp, 0, 0, c.width, c.height);
  return c;
}
function facePts(lmk, cv) {
  const r = lmk.detect(cv);
  if (!r.faceLandmarks || !r.faceLandmarks.length) return null;
  return r.faceLandmarks[0].map((p) => [p.x * cv.width, p.y * cv.height]);
}
function rollDeg(p) {
  const a = [(p[33][0] + p[133][0]) / 2, (p[33][1] + p[133][1]) / 2],
    b = [(p[362][0] + p[263][0]) / 2, (p[362][1] + p[263][1]) / 2];
  return (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;
}
function rotateBound(cv, deg) {
  const w = cv.width,
    h = cv.height,
    t = (deg * Math.PI) / 180,
    cos = Math.abs(Math.cos(t)),
    sin = Math.abs(Math.sin(t));
  const nw = Math.round(h * sin + w * cos),
    nh = Math.round(h * cos + w * sin),
    c = mk(nw, nh),
    g = c.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, nw, nh);
  g.translate(nw / 2, nh / 2);
  g.rotate(-t);
  g.drawImage(cv, -w / 2, -h / 2);
  return c;
}
async function personAlpha(cv) {
  const { model, proc } = await getSeg();
  const blob = await new Promise((r) => cv.toBlob(r, "image/png")),
    url = URL.createObjectURL(blob);
  try {
    const img = await RawImage.read(url),
      { pixel_values } = await proc(img),
      { output } = await model({ input: pixel_values });
    const m = await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
      img.width,
      img.height,
    );
    const o = new Float32Array(cv.width * cv.height);
    for (let i = 0; i < o.length; i++) o[i] = m.data[i] / 255;
    return o;
  } finally {
    URL.revokeObjectURL(url);
  }
}
function headGeo(p) {
  const top = Math.min(...IDX.oval.map((i) => p[i][1]));
  return {
    fw: Math.abs(p[EAR_R][0] - p[EAR_L][0]),
    fh: p[CHIN][1] - top,
    top,
    cx: (p[EAR_L][0] + p[EAR_R][0]) / 2,
  };
}
function cropBox(p, W, H) {
  const { fw, fh, top, cx } = headGeo(p);
  const x0 = Math.trunc(cx - CROP_SIDE * fw),
    x1 = Math.trunc(cx + CROP_SIDE * fw);
  const y0 = Math.trunc(top - CROP_ABOVE * fh),
    y1 = Math.trunc(p[CHIN][1] + CROP_BELOW_CHIN * fh);
  return {
    box: [Math.max(0, x0), Math.max(0, y0), Math.min(W, x1), Math.min(H, y1)],
    clipped: x0 < 0 || y0 < 0 || x1 > W,
  };
}

/* ------------------------------------------------------------------ head region (neck / collar removal) */
function headRegion(w, h, lp, fw, fh, keepHair) {
  const eL = lp[EAR_L],
    eR = lp[EAR_R],
    earY = (eL[1] + eR[1]) / 2,
    chinY = lp[CHIN][1],
    oc = mean2(IDX.oval.map((i) => lp[i]));
  const jaw0 = JAW.map((i) => lp[i].slice());
  const jaw = jaw0.map((q) => {
    const t = clamp((q[1] - earY) / Math.max(1, chinY - earY));
    return [
      oc[0] + (q[0] - oc[0]) * (1 + C.JAW_GROW_X),
      oc[1] + (q[1] - oc[1]) * (1 + C.JAW_GROW_Y) + t * C.JAW_DROP * fh,
    ];
  });
  const sideY = earY + (keepHair ? C.SIDE_DROP_HAIR : C.SIDE_DROP) * fh;
  let lobeY, xl, xr;
  if (keepHair) {
    lobeY = sideY;
    xl = 0;
    xr = w - 1;
  } else {
    lobeY = Math.max(sideY, lp[NOSE_BOTTOM][1] + C.EAR_LOBE * fh);
    const yaw = (lp[NOSE][0] - (eL[0] + eR[0]) / 2) / Math.max(1, fw);
    xl = Math.max(
      0,
      eL[0] - (C.EAR_OUT + Math.max(0, yaw) * C.EAR_YAW_BOOST) * fw,
    );
    xr = Math.min(
      w - 1,
      eR[0] + (C.EAR_OUT + Math.max(0, -yaw) * C.EAR_YAW_BOOST) * fw,
    );
  }
  lobeY = Math.min(lobeY, chinY - 0.1 * fh);
  let j = jaw.filter((q) => q[1] >= lobeY);
  if (j.length < 3) j = jaw0;
  return polyMask(w, h, [
    [
      [0, 0],
      [w - 1, 0],
      [w - 1, sideY],
      [xr, sideY],
      [xr, lobeY],
      ...j.slice().reverse(),
      [xl, lobeY],
      [xl, sideY],
      [0, sideY],
    ],
  ]);
}

/* ------------------------------------------------------------------ matting (guided filter) */
function matte(d, a, w, h) {
  const n = w * h,
    I = new Float32Array(n),
    ch = [0, 1, 2].map(() => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    const r = d[4 * i] / 255,
      g = d[4 * i + 1] / 255,
      b = d[4 * i + 2] / 255;
    ch[0][i] = r;
    ch[1][i] = g;
    ch[2][i] = b;
    I[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  const r = Math.max(4, Math.round(0.012 * Math.max(w, h))),
    eps = 1e-3;
  const mI = box(I, w, h, r),
    mp = box(a, w, h, r),
    Ip = new Float32Array(n),
    II = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    Ip[i] = I[i] * a[i];
    II[i] = I[i] * I[i];
  }
  const cIp = box(Ip, w, h, r),
    cII = box(II, w, h, r),
    A = new Float32Array(n),
    B = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    A[i] = (cIp[i] - mI[i] * mp[i]) / (cII[i] - mI[i] * mI[i] + eps);
    B[i] = mp[i] - A[i] * mI[i];
  }
  const mA = box(A, w, h, r),
    mB = box(B, w, h, r);
  const hi = new Float32Array(n),
    lo = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    hi[i] = a[i] > 0.92 ? 1 : 0;
    lo[i] = a[i] < 0.04 ? 1 : 0;
  }
  const rk = Math.max(1, Math.max(3, Math.trunc(0.012 * Math.max(w, h))) >> 1);
  const fg = ero(hi, w, h, rk),
    bg = ero(lo, w, h, rk),
    al = new Float32Array(n),
    un = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    al[i] = fg[i] ? 1 : bg[i] ? 0 : clamp((mA[i] * I[i] + mB[i] - 0.05) / 0.9);
    un[i] = al[i] < 0.985 ? 1 : 0;
  }
  // colour decontamination: edge pixels take colour from nearby solid-foreground pixels
  const rr = Math.max(6, 3 * r),
    den = box(fg, w, h, rr),
    F = ch.map((c) => {
      const m = new Float32Array(n);
      for (let i = 0; i < n; i++) m[i] = c[i] * fg[i];
      return box(m, w, h, rr);
    });
  const edge = gauss(un, w, h, 2),
    out = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++)
    for (let k = 0; k < 3; k++) {
      const f = den[i] > 1e-3 ? F[k][i] / den[i] : ch[k][i];
      out[4 * i + k] = (ch[k][i] * (1 - edge[i]) + f * edge[i]) * 255;
    }
  return { al, out };
}

/* ------------------------------------------------------------------ checks + metadata */
function sampleSkin(rgb, a, w, h, pts) {
  const vals = [];
  for (const i of [CHEEK_L, CHEEK_R, FOREHEAD, 205, 425]) {
    const cx = Math.trunc(pts[i][0]),
      cy = Math.trunc(pts[i][1]),
      px = [[], [], []];
    for (let y = Math.max(0, cy - 4); y <= Math.min(h - 1, cy + 4); y++)
      for (let x = Math.max(0, cx - 4); x <= Math.min(w - 1, cx + 4); x++)
        if (a[y * w + x] > 0.9)
          for (let k = 0; k < 3; k++) px[k].push(rgb[4 * (y * w + x) + k]);
    if (px[0].length)
      vals.push(
        px.map((v) => {
          v.sort((p, q) => p - q);
          return v[v.length >> 1];
        }),
      );
  }
  if (!vals.length) return [224, 172, 140];
  return [0, 1, 2].map((k) => {
    const v = vals.map((x) => x[k]).sort((p, q) => p - q);
    return Math.round(v[v.length >> 1]);
  });
}
function buildChecks(al, w, h, lp, fw2, fh2, clipped) {
  const inside = faceCore(w, h, lp, 0.85);
  let cnt = 0,
    sum = 0,
    holes = 0;
  for (let i = 0; i < al.length; i++)
    if (inside[i] > 0.5) {
      cnt++;
      sum += al[i];
      if (al[i] < 0.5) holes++;
    }
  const solid = cnt ? sum / cnt : 0,
    holeFrac = cnt ? holes / cnt : 1;
  const cx = Math.min(w - 1, Math.max(0, Math.trunc(lp[CHIN][0]))),
    cy = Math.min(h - 1, Math.max(0, Math.trunc(lp[CHIN][1])));
  const neckY = Math.trunc(cy + (C.JAW_DROP + 0.12) * fh2);
  let neckOk = neckY >= h;
  if (!neckOk) {
    neckOk = true;
    for (let i = neckY * w; i < al.length; i++)
      if (al[i] >= 0.05) {
        neckOk = false;
        break;
      }
  }
  const bin = new Uint8Array(w * h);
  for (let i = 0; i < bin.length; i++) bin[i] = al[i] > 0.5 ? 1 : 0;
  const { sizes, n } = comps(bin, w, h);
  let blobs = 0;
  for (let i = 1; i <= n; i++) if (sizes[i] >= 0.002 * w * h) blobs++;
  return [
    {
      name: "Face fully opaque (no holes)",
      ok: solid > 0.985 && holeFrac < 0.005,
    },
    { name: "Single connected head (no background remnants)", ok: blobs === 1 },
    { name: "Chin intact", ok: al[cy * w + cx] > 0.9 },
    { name: "Neck / clothing removed", ok: neckOk },
    { name: "Hair not cut off by photo border", ok: !clipped },
    { name: "Enough resolution (face >= 140 px wide)", ok: fw2 >= 140 },
  ];
}
function buildMeta(lp, rgb, al, w, h, fw2, fh2) {
  const P = (ids) =>
      ids.map((i) => [
        Math.round(lp[i][0] * 10) / 10,
        Math.round(lp[i][1] * 10) / 10,
      ]),
    meta = {};
  for (const k in IDX) meta[k] = P(IDX[k]);
  Object.assign(meta, {
    nose: P([NOSE])[0],
    chin: P([CHIN])[0],
    forehead: P([FOREHEAD])[0],
    bridge: P([BRIDGE])[0],
    cheekL: P([CHEEK_L])[0],
    cheekR: P([CHEEK_R])[0],
    earL: P([EAR_L])[0],
    earR: P([EAR_R])[0],
    mouthL: P([61])[0],
    mouthR: P([291])[0],
    faceW: Math.round(fw2 * 10) / 10,
    faceH: Math.round(fh2 * 10) / 10,
    skin: sampleSkin(rgb, al, w, h, lp),
    size: [w, h],
  });
  return meta;
}

/* ------------------------------------------------------------------ pipeline */
async function run(file, keepHair = false, onStage = () => {}) {
  onStage(1, "Analyzing photograph");
  await tick();
  let cv = await loadImage(file);

  onStage(2, "Detecting head and 478 facial landmarks");
  await tick();
  const lmk = await getLandmarker();
  let pts = facePts(lmk, cv);
  if (!pts) throw new Error("No face found. Use a clear, front-facing photo.");
  const roll = rollDeg(pts);
  if (Math.abs(roll) > 1.5 && Math.abs(roll) <= 35) {
    cv = rotateBound(cv, roll);
    pts = facePts(lmk, cv);
    if (!pts)
      throw new Error("Face lost after levelling the head. Try another photo.");
  }

  onStage(3, "Segmenting head and hair");
  await tick();
  const alpha = await personAlpha(cv);

  const W = cv.width,
    H = cv.height,
    { fw, fh } = headGeo(pts),
    {
      box: [x0, y0, x1, y1],
      clipped,
    } = cropBox(pts, W, H);
  const w = x1 - x0,
    h = y1 - y0;
  if (w < 8 || h < 8) throw new Error("Head extraction failed. Crop is empty.");
  const crop = mk(w, h),
    cg = crop.getContext("2d");
  cg.drawImage(cv, x0, y0, w, h, 0, 0, w, h);
  const rgba = cg.getImageData(0, 0, w, h).data,
    a0 = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) a0[y * w + x] = alpha[(y + y0) * W + x + x0];
  let lp = pts.map((p) => [p[0] - x0, p[1] - y0]);

  onStage(4, "Matting hair strands and refining transparent edges");
  await tick();
  const core = faceCore(w, h, lp, 0.9);
  let bad = 0,
    cc = 0;
  for (let i = 0; i < core.length; i++)
    if (core[i] > 0.5) {
      cc++;
      if (a0[i] < 0.5) bad++;
    }
  const coreBad = bad / Math.max(1, cc);
  const region = headRegion(w, h, lp, fw, fh, keepHair);
  const regionPre = dil(
    region,
    w,
    h,
    Math.max(3, Math.trunc(0.01 * fw) | 1) >> 1,
  );
  let a = new Float32Array(w * h);
  for (let i = 0; i < a.length; i++) a[i] = Math.max(a0[i], core[i]);
  const kf = keepFace(a, w, h, lp[NOSE], 0.02 * fw);
  for (let i = 0; i < a.length; i++) a[i] *= kf[i] * regionPre[i];
  const { al, out } = matte(rgba, a, w, h);

  onStage(5, "Cutting neck and cleaning background remnants");
  await tick();
  const soft = gauss(region, w, h, Math.max(1, C.FEATHER * fw));
  for (let i = 0; i < al.length; i++) al[i] *= soft[i];
  const kf2 = keepFace(al, w, h, lp[NOSE], 0.01 * fw);
  for (let i = 0; i < al.length; i++) if (kf2[i] <= 0.02) al[i] = 0;

  let mnx = w,
    mny = h,
    mxx = -1,
    mxy = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (al[y * w + x] > 0.02) {
        if (x < mnx) mnx = x;
        if (x > mxx) mxx = x;
        if (y < mny) mny = y;
        if (y > mxy) mxy = y;
      }
  if (mxx < 0)
    throw new Error("Head extraction failed. No foreground pixels detected.");
  const pad = 4,
    bx0 = Math.max(0, mnx - pad),
    bx1 = Math.min(w, mxx + pad),
    by0 = Math.max(0, mny - pad),
    by1 = Math.min(h, mxy + pad);
  const bw = bx1 - bx0,
    bh = by1 - by0,
    sc = Math.min(1, OUT_MAXSIDE / Math.max(bw, bh)),
    nw = Math.max(1, Math.round(bw * sc)),
    nh = Math.max(1, Math.round(bh * sc));

  for (let i = 0; i < al.length; i++) out[4 * i + 3] = al[i] * 255;
  const full = mk(w, h);
  full.getContext("2d").putImageData(new ImageData(out, w, h), 0, 0);
  const fin = mk(nw, nh),
    fg = fin.getContext("2d");
  fg.imageSmoothingQuality = "high";
  fg.drawImage(full, bx0, by0, bw, bh, 0, 0, nw, nh);
  lp = lp.map((p) => [(p[0] - bx0) * sc, (p[1] - by0) * sc]);
  const fw2 = fw * sc,
    fh2 = fh * sc;

  onStage(6, "Running quality checks");
  await tick();
  const px = fg.getImageData(0, 0, nw, nh).data,
    al2 = new Float32Array(nw * nh);
  for (let i = 0; i < al2.length; i++) al2[i] = px[4 * i + 3] / 255;
  const checks = buildChecks(al2, nw, nh, lp, fw2, fh2, clipped),
    info = [];
  if (coreBad > 0.01)
    info.push(
      "Segmentation had holes inside the face (glasses/reflections). They were repaired from landmarks.",
    );
  return {
    done: true,
    png: fin.toDataURL("image/png"),
    landmarks: buildMeta(lp, px, al2, nw, nh, fw2, fh2),
    checks,
    info,
    passed: checks.every((c) => c.ok),
  };
}

/* warm the models in the background so the first extraction is faster */
const warm = () =>
  Promise.all([getLandmarker(), getSeg()]).catch((e) =>
    console.warn("model warm-up failed", e),
  );
window.HeadExtract = { run, warm };
warm();
