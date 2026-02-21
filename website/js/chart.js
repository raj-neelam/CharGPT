// chart.js — Multi-chart: Loss (WMA + Raw) + Test Loss + Learning Rate

class LossChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) { console.warn('[Chart] Canvas not found:', canvasId); return; }
        this.wmaData = [];
        this.rawData = [];
        this.testData = [];      // (step, loss) at epoch boundaries
        this.lrData = [];      // (step, lr) at each batch
        this.labels = [];
        this.epochLines = [];
        this._build();
    }

    _colors() {
        const dark = document.documentElement.getAttribute('data-theme') !== 'light';
        return {
            text: dark ? '#8b92b4' : '#5b6690',
            grid: dark ? '#2a3050' : '#e2e8f0',
            wma: '#5b8ef0',
            raw: dark ? 'rgba(91,142,240,0.18)' : 'rgba(59,106,232,0.12)',
            test: '#34c98e',
            lr: '#f0b429',
            epoch: dark ? 'rgba(240,180,41,0.5)' : 'rgba(176,122,16,0.5)',
        };
    }

    _build() {
        const c = this._colors();
        const baseScaleOpts = (axId, pos) => ({
            [axId]: {
                type: 'linear',
                position: pos,
                ticks: { color: c.text, font: { size: 9 }, maxTicksLimit: 6 },
                grid: { color: axId === 'y' ? c.grid : 'transparent' },
                // Y-axis auto-ranges from actual data — no forced 0
            }
        });

        this.chart = new Chart(this.canvas, {
            type: 'line',
            data: {
                labels: this.labels,
                datasets: [
                    {
                        label: 'WMA Loss',
                        data: this.wmaData,
                        borderColor: c.wma,
                        backgroundColor: 'transparent',
                        borderWidth: 1.8,
                        pointRadius: 0,
                        tension: 0.3,
                        yAxisID: 'y',
                        order: 1,
                    },
                    {
                        label: 'Raw Loss',
                        data: this.rawData,
                        borderColor: c.raw,
                        backgroundColor: 'transparent',
                        borderWidth: 0.7,
                        pointRadius: 0,
                        tension: 0.1,
                        yAxisID: 'y',
                        order: 2,
                    },
                    {
                        label: 'Test Loss',
                        data: this.testData,
                        borderColor: c.test,
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBorderWidth: 0,
                        pointBackgroundColor: c.test,
                        tension: 0.2,
                        yAxisID: 'y',
                        spanGaps: true,
                        order: 0,
                    },
                    {
                        label: 'LR',
                        data: this.lrData,
                        borderColor: c.lr,
                        backgroundColor: 'transparent',
                        borderWidth: 1.4,
                        pointRadius: 0,
                        tension: 0,
                        yAxisID: 'y2',
                        order: 3,
                        borderDash: [3, 3],
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        labels: {
                            color: c.text,
                            font: { size: 9.5 },
                            boxWidth: 16,
                            padding: 6,
                            usePointStyle: true,
                        },
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => {
                                const v = ctx.parsed.y;
                                if (v === null || v === undefined) return null;
                                return `${ctx.dataset.label}: ${typeof v === 'number' ? v.toFixed(4) : v}`;
                            },
                        },
                    },
                },
                scales: {
                    x: {
                        ticks: { color: c.text, font: { size: 9 }, maxTicksLimit: 8 },
                        grid: { color: c.grid },
                    },
                    y: {
                        position: 'left',
                        ticks: { color: c.text, font: { size: 9 }, maxTicksLimit: 6 },
                        grid: { color: c.grid },
                        // Auto-range: no min:0 — Chart.js will fit to data
                    },
                    y2: {
                        position: 'right',
                        ticks: { color: c.lr, font: { size: 9 }, maxTicksLimit: 4 },
                        grid: { drawOnChartArea: false },
                        title: { display: false },
                    },
                },
            },
        });
    }

    addPoint(step, wmaLoss, rawLoss, lr = null) {
        this.labels.push(step);
        this.wmaData.push(wmaLoss);
        this.rawData.push(rawLoss);
        // Pad test and LR datasets with null to keep index alignment
        this.testData.push(null);
        this.lrData.push(lr !== null ? lr : (this.lrData.at(-1) ?? null));
        if (step % 5 === 0 || step < 10) this.chart.update('none');
    }

    addTestPoint(step, testLoss) {
        // Update the last entry in testData array to this loss at this step
        // Find matching label index
        const idx = this.labels.lastIndexOf(step);
        if (idx >= 0) {
            this.testData[idx] = testLoss;
        } else {
            // Append
            this.labels.push(step);
            this.wmaData.push(null);
            this.rawData.push(null);
            this.testData.push(testLoss);
            this.lrData.push(null);
        }
        this.chart.update('none');
    }

    addEpochMarker(step) {
        this.epochLines.push(step);
        this.chart.update('none');
    }

    reset() {
        this.wmaData = [];
        this.rawData = [];
        this.testData = [];
        this.lrData = [];
        this.labels = [];
        this.epochLines = [];
        if (this.chart) {
            this.chart.data.labels = this.labels;
            this.chart.data.datasets[0].data = this.wmaData;
            this.chart.data.datasets[1].data = this.rawData;
            this.chart.data.datasets[2].data = this.testData;
            this.chart.data.datasets[3].data = this.lrData;
            this.chart.update('none');
        }
    }

    destroy() {
        if (this.chart) { this.chart.destroy(); this.chart = null; }
    }
}

window.LossChart = LossChart;
