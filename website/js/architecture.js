// architecture.js — GPT Decoder Architecture (v4)
// Flow direction: BOTTOM (input) → TOP (output), arrows point UPWARD

'use strict';

class ArchitectureVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.cfg = {
      vocabSize: 70, embDim: 64, contextLength: 128,
      numLayers: 2, numHeads: 4, dropoutRate: 0.1, ffMul: 2,
    };
    if (!this.container) return;
    this.render();
  }

  update(cfg) { this.cfg = { ...this.cfg, ...cfg }; this.render(); }

  _dark() { return document.documentElement.getAttribute('data-theme') !== 'light'; }

  render() {
    if (!this.container) return;
    const dark = this._dark();
    const C = {
      bg: dark ? '#0f1117' : '#f8faff',
      surface: dark ? '#1c2238' : '#ffffff',
      border: dark ? '#2a3050' : '#d0d8ee',
      accent: dark ? '#5b8ef0' : '#3b6ae8',
      accentDim: dark ? '#3a5cb888' : '#bdcef888',
      gold: dark ? '#c9a84c' : '#a07818',
      goldDim: dark ? '#c9a84ccc' : '#a07818cc',
      green: dark ? '#34c98e' : '#1da068',
      purple: dark ? '#9b72cf' : '#7040c0',
      text: dark ? '#dde4f8' : '#1a2040',
      sub: dark ? '#8b92b4' : '#5060a0',
      dim: dark ? '#3a4470' : '#aabcd8',
      arr: dark ? '#5b8ef0' : '#3b6ae8',
      res: dark ? '#f0b429' : '#c08800',
      blockBdr: dark ? '#f0b42970' : '#d09030aa',
    };

    const W = 280;
    const cx = W / 2;
    const BW = 206;
    const bx = cx - BW / 2;
    const BH = 30;
    const A = 24;   // arrow length between blocks
    const els = [];

    // ── Helpers ────────────────────────────────────────────────────────────

    // Upward arrow: from y1 (bottom) to y2 (top), using SVG coord system
    // In SVG y increases downward, so "upward visual arrow" means arrow at lower y value
    // We draw from (x, y1_svg) UP to (x, y2_svg) where y2_svg < y1_svg
    const arrowUp = (x, fromY, toY, col = C.arr) => {
      const tip = toY;
      const base = fromY;
      return [
        `<line x1="${x}" y1="${base}" x2="${x}" y2="${tip + 7}"
                  stroke="${col}" stroke-width="1.4"/>`,
        `<polygon points="${x},${tip} ${x - 4},${tip + 9} ${x + 4},${tip + 9}"
                  fill="${col}"/>`,
      ].join('');
    };

    const box = (x, y, w, h, fill, stroke, label, sub = null, opts = {}) => {
      const rx = opts.rx ?? 7;
      const fs = opts.fs ?? 11;
      const fw = opts.fw ?? '600';
      const opacity = opts.opacity ?? 1;
      const ty = sub ? y + h / 2 - 4 : y + h / 2 + 4;
      const col = opts.textCol ?? C.text;
      return [
        `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}"
                  fill="${fill}" stroke="${stroke}" stroke-width="1.3" opacity="${opacity}"/>`,
        `<text x="${x + w / 2}" y="${ty}" text-anchor="middle"
                  dominant-baseline="auto" fill="${col}"
                  font-size="${fs}" font-weight="${fw}"
                  font-family="Inter,sans-serif">${label}</text>`,
        sub ? `<text x="${x + w / 2}" y="${y + h / 2 + 11}"
                  text-anchor="middle" fill="${C.sub}" font-size="8.5"
                  font-family="Inter,sans-serif">${sub}</text>` : '',
      ].join('');
    };

    const addNorm = (y) =>
      box(bx + 10, y, BW - 20, BH - 2, C.surface, C.accentDim,
        'Add & Layer Norm', null, { rx: 13, fs: 10, fw: '500' });

    // Residual skip connection — arrowhead points INWARD toward Add & Layer Norm
    // tapY: where skip taps (layer input, visually lower in SVG)
    // joinY: where it merges at Add&Norm center (visually higher)
    // side: 'left' | 'right'
    const residual = (side, tapY, joinY) => {
      const rx = side === 'left' ? bx - 22 : bx + BW + 22;
      const jx = side === 'left' ? bx + 10 : bx + BW - 10;
      // Arrow points INWARD (toward the box center)
      const arrDir = side === 'left' ? 1 : -1;
      const joinMid = joinY;
      return [
        // dot where skip begins
        `<circle cx="${rx}" cy="${tapY}" r="3.5" fill="${C.res}" opacity="0.9"/>`,
        // vertical dashed line going up
        `<line x1="${rx}" y1="${tapY}" x2="${rx}" y2="${joinMid + 5}"
                  stroke="${C.res}" stroke-width="1.5" stroke-dasharray="5,3" opacity="0.88"/>`,
        // horizontal jog toward AddNorm
        `<line x1="${rx}" y1="${joinMid}" x2="${jx}" y2="${joinMid}"
                  stroke="${C.res}" stroke-width="1.5" opacity="0.88"/>`,
        // arrowhead pointing INWARD (toward box)
        `<polygon points="${jx},${joinMid}
                  ${jx + arrDir * 9},${joinMid - 4}
                  ${jx + arrDir * 9},${joinMid + 4}"
                  fill="${C.res}" opacity="0.88"/>`,
      ].join('');
    };

    // ── Layout — rendered TOP→BOTTOM in SVG, but arrows point UPWARD ──────
    // We lay out from top (output: Softmax) to bottom (input: Embedding)
    // Arrows point from lower element to upper element (upward)

    const { numLayers, numHeads, embDim, ffMul, contextLength, vocabSize } = this.cfg;
    let y = 14;

    /* ── Softmax ── */
    const smW = BW - 70;
    els.push(box(cx - smW / 2, y, smW, BH - 2, C.green, C.green,
      'Softmax', null, { fs: 12, opacity: 1.0 }));
    y += BH - 2;
    // ↑ arrow from Linear BELOW to Softmax ABOVE
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── Linear (Unembed) ── */
    els.push(box(bx + 8, y, BW - 16, BH, C.surface, C.accent,
      'Linear (Unembed)', null, { fs: 10 }));
    y += BH;
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ══ DECODER BLOCK ══ */
    const blockTopY = y - 4;

    /* ── Add & Norm (top — FFN residual merge) ── */
    const AN_ffn_top = y;
    els.push(addNorm(y));
    y += BH - 2;
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── FFN ── */
    const FFN_top = y;
    const ffnH = 42;
    els.push(box(bx + 2, y, BW - 4, ffnH, C.purple, C.purple,
      `FFN (×${ffMul})`, `hidden dim = embDim × ${ffMul}  ReLU`,
      { fs: 10, rx: 8, opacity: 0.96 }));
    y += ffnH;
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── Add & Norm (FFN residual enters here — this is after MHA block) ── */
    const AN_ffn_bot = y;
    els.push(addNorm(y));
    y += BH - 2;

    /* RESIDUAL around FFN:
       - Skip taps at AN_ffn_bot (MHA output / FFN input), LEFT side
       - Skip merges into AN_ffn_top (FFN output AddNorm) center
    */
    const res_ffn_tapY = AN_ffn_bot + (BH - 2) / 2;
    const res_ffn_joinY = AN_ffn_top + (BH - 2) / 2;
    els.push(residual('left', res_ffn_tapY, res_ffn_joinY));

    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── Add & Norm (MHA residual merge) ── */
    const AN_mha_top = y;
    els.push(addNorm(y));
    y += BH - 2;
    els.push(arrowUp(cx, y + A - 2, y, C.arr));
    y += A - 2;

    /* ── Masked Multi-Head Attention block ── */
    const MHA_top = y;
    const numH = Math.min(numHeads, 5);
    const cardW = 34, cardH = 22;
    const xGap = 5, yStep = 7;
    const groupW = numH * cardW + (numH - 1) * xGap + (numH - 1) * yStep;
    const groupH = cardH + (numH - 1) * yStep;
    const mhaBoxH = groupH + 52;   // header + QKV row + padding
    const mhaBoxW = BW;

    els.push(
      `<rect x="${bx}" y="${MHA_top}" width="${mhaBoxW}" height="${mhaBoxH}"
              rx="9" fill="${C.surface}" stroke="${C.accent}" stroke-width="1.2"
              opacity="0.92"/>`,
      `<text x="${cx}" y="${MHA_top + 14}" text-anchor="middle"
              fill="${C.accent}" font-size="9.5" font-weight="600"
              font-family="Inter,sans-serif">Masked Multi-Head Attention</text>`
    );

    // Staggered head cards — front card (head 0) at top-left, receding to right-bottom
    const gStartX = cx - groupW / 2 + (numH - 1) * yStep;
    const gStartY = MHA_top + 20;
    for (let i = numH - 1; i >= 0; i--) {
      // Draw back cards first (i = numH-1 is furthest back)
      const hx = gStartX + i * (cardW + xGap) - i * yStep;
      const hy = gStartY + i * yStep;
      const op = 0.35 + ((numH - 1 - i) / Math.max(numH - 1, 1)) * 0.65;
      els.push(
        `<rect x="${hx}" y="${hy}" width="${cardW}" height="${cardH}"
                  rx="5" fill="${C.accentDim}" stroke="${C.accent}"
                  stroke-width="1" opacity="${op.toFixed(2)}"/>`,
        `<text x="${hx + cardW / 2}" y="${hy + cardH / 2 + 4}"
                  text-anchor="middle" fill="${C.text}"
                  font-size="8" font-weight="500"
                  font-family="Inter,sans-serif">Attn</text>`
      );
    }

    // Q K V arrows + labels
    const qkvBaseY = gStartY + groupH + 6;
    [['Q', -24], ['K', 0], ['V', 24]].forEach(([l, dx]) => {
      const lx = cx + dx;
      els.push(
        `<line x1="${lx}" y1="${qkvBaseY + 14}" x2="${lx}" y2="${qkvBaseY + 6}"
                  stroke="${C.dim}" stroke-width="1.2" marker-end="url(#arr-marker)"/>`,
        `<text x="${lx}" y="${qkvBaseY + 26}" text-anchor="middle"
                  fill="${C.sub}" font-size="9" font-weight="700"
                  font-family="Inter,monospace">${l}</text>`
      );
    });

    const MHA_bot = MHA_top + mhaBoxH;
    y = MHA_bot;
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── Add & Norm (block bottom — decoder input residual merge) ── */
    const AN_mha_bot = y;
    els.push(addNorm(y));
    y += BH - 2;

    /* RESIDUAL around MHA:
       - Skip taps at AN_mha_bot (decoder block INPUT), RIGHT side
       - Skip merges at AN_mha_top (MHA output AddNorm) center
    */
    const res_mha_tapY = AN_mha_bot + (BH - 2) / 2;
    const res_mha_joinY = AN_mha_top + (BH - 2) / 2;
    els.push(residual('right', res_mha_tapY, res_mha_joinY));

    /* ══ DECODER BLOCK outline ══ */
    const blockBotY = y + 6;
    const blockH = blockBotY - blockTopY;
    // Insert behind boxes
    els.unshift(
      `<rect x="${bx - 10}" y="${blockTopY}" width="${BW + 20}" height="${blockH}"
              rx="12" fill="none" stroke="${C.blockBdr}" stroke-width="1.5"
              stroke-dasharray="6,4"/>`,
      `<text x="${bx - 10}" y="${blockTopY - 5}"
              fill="${C.blockBdr}" font-size="9.5" font-weight="600"
              font-family="Inter,sans-serif">Decoder Block ×${numLayers}</text>`
    );

    els.push(arrowUp(cx, y + A + 4, y, C.arr));
    y += A + 4;

    /* ── Positional Encoding — SVG visualization ── */
    const peY = y;
    const peW = BW;
    const peH = 46;
    // Draw a mini sinusoidal PE grid — 3 rows × 8 cols
    const cols = 8, rows = 3;
    const cellW = (peW - 20) / cols;
    const cellH = 9;
    const peGx = bx + 10;
    const peGy = peY + 16;

    els.push(
      `<rect x="${bx}" y="${peY}" width="${peW}" height="${peH}"
              rx="8" fill="${C.surface}" stroke="${C.gold}" stroke-width="1.2"
              opacity="0.88"/>`,
      `<text x="${cx}" y="${peY + 13}" text-anchor="middle"
              fill="${C.gold}" font-size="9.5" font-weight="600"
              font-family="Inter,sans-serif">Positional Encoding</text>`
    );
    // Mini sinusoidal heatmap grid
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = Math.sin(c / Math.pow(10000, (2 * r) / embDim));
        const normalized = (val + 1) / 2; // 0..1
        const alpha = (0.15 + normalized * 0.7).toFixed(2);
        const fill = dark
          ? `rgba(201,168,76,${alpha})`
          : `rgba(160,120,24,${alpha})`;
        els.push(
          `<rect x="${peGx + c * cellW}" y="${peGy + r * (cellH + 1)}"
                      width="${cellW - 1}" height="${cellH}"
                      rx="1.5" fill="${fill}"/>`
        );
      }
    }

    y = peY + peH;
    els.push(arrowUp(cx, y + A, y, C.arr));
    y += A;

    /* ── Embedding Vectors (heatmap, same design as PE) ── */
    const embH = 46;
    const embCols = 8, embRows = 3;
    const embCellW = (BW - 20) / embCols;
    const embCellH = 9;
    const embGx = bx + 10;
    const embGy = y + 16;

    els.push(
      `<rect x="${bx}" y="${y}" width="${BW}" height="${embH}"
              rx="8" fill="${C.surface}" stroke="${C.gold}" stroke-width="1.2"/>`,
      `<text x="${cx}" y="${y + 13}" text-anchor="middle"
              fill="${C.gold}" font-size="9.5" font-weight="600"
              font-family="Inter,sans-serif">Embedding  Vocab=${vocabSize}  dim=${embDim}</text>`
    );
    // Mini embedding heatmap — simulate learned token vectors (column = dim, row = token class)
    for (let r = 0; r < embRows; r++) {
      for (let c = 0; c < embCols; c++) {
        // Use a quasi-random pattern to simulate learned weights
        const val = Math.sin((r * 3 + c) * 0.7 + c / (embCols * 0.5));
        const normalized = (val + 1) / 2;
        const alpha = (0.12 + normalized * 0.72).toFixed(2);
        const fill = dark
          ? `rgba(201,168,76,${alpha})`
          : `rgba(160,120,24,${alpha})`;
        els.push(
          `<rect x="${embGx + c * embCellW}" y="${embGy + r * (embCellH + 1)}"
                      width="${embCellW - 1}" height="${embCellH}"
                      rx="1.5" fill="${fill}"/>`
        );
      }
    }
    y += embH + 5;

    // context / heads / ff info
    els.push(
      `<text x="${cx}" y="${y + 11}" text-anchor="middle"
              fill="${C.dim}" font-size="8" font-family="Inter,sans-serif">
              ctx=${contextLength}  heads=${numHeads}  ff×${ffMul}  drop=${(this.cfg.dropoutRate).toFixed(2)}</text>`
    );
    y += 20;

    // Input arrow from below
    els.push(arrowUp(cx, y + 18, y, C.dim));
    y += 20;

    /* ── Assemble ── */
    const totalH = y + 4;
    const defsStr = `<defs>
          <marker id="arr-marker" markerWidth="6" markerHeight="6"
            refX="3" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 Z" fill="${C.dim}"/>
          </marker>
        </defs>`;
    this.container.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 ${W} ${totalH}" width="${W}" height="${totalH}"
          style="display:block;max-width:100%;height:auto;">
          <rect width="${W}" height="${totalH}" fill="${C.bg}" rx="8"/>
          ${defsStr}
          ${els.join('\n')}
        </svg>`;
  }
}

window.ArchitectureVisualizer = ArchitectureVisualizer;
