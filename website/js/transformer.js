// transformer.js — GPT Decoder-only Transformer (TF.js)
// Character-level language model with configurable FFN multiplier

/**
 * Custom TF.js Layer for causal self-attention
 * Input: QKV tensor [batch, seq, 3*embDim]
 * Output: [batch, seq, embDim]
 */
class CausalSelfAttentionLayer extends tf.layers.Layer {
    constructor(config) {
        super(config);
        this.numHeads = config.numHeads;
        this.headDim = config.headDim;
    }

    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1], this.numHeads * this.headDim];
    }

    call(inputs) {
        return tf.tidy(() => {
            const qkv = Array.isArray(inputs) ? inputs[0] : inputs;
            const embDim = this.numHeads * this.headDim;
            const batchSize = qkv.shape[0] || 1;
            const seqLen = qkv.shape[1];

            const Q = tf.slice(qkv, [0, 0, 0], [-1, -1, embDim]);
            const K = tf.slice(qkv, [0, 0, embDim], [-1, -1, embDim]);
            const V = tf.slice(qkv, [0, 0, embDim * 2], [-1, -1, embDim]);

            const reshape = (t) =>
                tf.transpose(
                    tf.reshape(t, [batchSize, seqLen, this.numHeads, this.headDim]),
                    [0, 2, 1, 3]
                );

            const Qr = reshape(Q), Kr = reshape(K), Vr = reshape(V);
            const scale = Math.sqrt(this.headDim);
            const scores = tf.div(tf.matMul(Qr, Kr, false, true), scale);

            const maskLower = tf.linalg.bandPart(tf.ones([seqLen, seqLen]), -1, 0);
            const maskUpper = tf.sub(tf.scalar(1.0), maskLower);
            const maskedScores = tf.sub(scores, tf.mul(maskUpper, tf.scalar(1e9)));

            const attnWeights = tf.softmax(maskedScores, -1);
            const attnOut = tf.matMul(attnWeights, Vr);
            const merged = tf.reshape(
                tf.transpose(attnOut, [0, 2, 1, 3]),
                [batchSize, seqLen, embDim]
            );
            return merged;
        });
    }

    getConfig() {
        return { ...super.getConfig(), numHeads: this.numHeads, headDim: this.headDim };
    }

    static get className() { return 'CausalSelfAttentionLayer'; }
}

tf.serialization.registerClass(CausalSelfAttentionLayer);

/**
 * Build one GPT decoder block
 */
function buildDecoderBlock(x, embDim, numHeads, dropoutRate, ffMul, blockIdx) {
    const headDim = embDim / numHeads;
    const ffUnits = embDim * ffMul;

    // Pre-LN → MHA → Residual
    const residual1 = x;
    x = tf.layers.layerNormalization({ name: `ln1_b${blockIdx}` }).apply(x);
    const qkv = tf.layers.dense({ units: embDim * 3, useBias: false, name: `qkv_b${blockIdx}` }).apply(x);
    let attnOut = new CausalSelfAttentionLayer({ numHeads, headDim, name: `csa_b${blockIdx}` }).apply(qkv);
    attnOut = tf.layers.dense({ units: embDim, useBias: false, name: `attn_out_b${blockIdx}` }).apply(attnOut);
    if (dropoutRate > 0) attnOut = tf.layers.dropout({ rate: dropoutRate, name: `attn_drop_b${blockIdx}` }).apply(attnOut);
    x = tf.layers.add({ name: `add1_b${blockIdx}` }).apply([residual1, attnOut]);

    // Pre-LN → FFN → Residual
    const residual2 = x;
    x = tf.layers.layerNormalization({ name: `ln2_b${blockIdx}` }).apply(x);
    x = tf.layers.dense({ units: ffUnits, activation: 'relu', name: `ffn1_b${blockIdx}` }).apply(x);
    x = tf.layers.dense({ units: embDim, name: `ffn2_b${blockIdx}` }).apply(x);
    if (dropoutRate > 0) x = tf.layers.dropout({ rate: dropoutRate, name: `ffn_drop_b${blockIdx}` }).apply(x);
    x = tf.layers.add({ name: `add2_b${blockIdx}` }).apply([residual2, x]);

    return x;
}

/**
 * Build CharacterGPT model
 * Inputs: [tokenInput [batch, seq], posInput [batch, seq]]
 * Output: logits [batch, seq, vocabSize]
 */
function buildTransformerModel(config) {
    const { vocabSize, embDim, contextLength, numLayers, numHeads, dropoutRate, ffMul = 4 } = config;

    if (embDim % numHeads !== 0) {
        throw new Error(`embDim (${embDim}) must be divisible by numHeads (${numHeads})`);
    }

    console.log(`🏗️ [Build] vocab=${vocabSize}, emb=${embDim}, ctx=${contextLength}, layers=${numLayers}, heads=${numHeads}, ffMul=${ffMul}, drop=${dropoutRate}`);

    const input = tf.input({ shape: [null], dtype: 'int32', name: 'token_input' });
    const posInput = tf.input({ shape: [null], dtype: 'int32', name: 'pos_input' });

    let x = tf.layers.embedding({ inputDim: vocabSize, outputDim: embDim, name: 'tok_embed' }).apply(input);
    const posE = tf.layers.embedding({ inputDim: contextLength, outputDim: embDim, name: 'pos_embed' }).apply(posInput);
    x = tf.layers.add({ name: 'add_embed' }).apply([x, posE]);

    if (dropoutRate > 0) x = tf.layers.dropout({ rate: dropoutRate, name: 'embed_drop' }).apply(x);

    for (let i = 0; i < numLayers; i++) {
        x = buildDecoderBlock(x, embDim, numHeads, dropoutRate, ffMul, i);
    }

    x = tf.layers.layerNormalization({ name: 'final_ln' }).apply(x);
    const logits = tf.layers.dense({ units: vocabSize, name: 'unembed' }).apply(x);

    const model = tf.model({ inputs: [input, posInput], outputs: logits, name: 'CharacterGPT' });
    console.log(`🏗️ [Build] Model ready — ${model.countParams().toLocaleString()} parameters`);
    return model;
}

/**
 * Generate text character-by-character with streaming callback
 */
async function generateText(model, tokenizer, startText, generateLen, temperature, contextLength, onChar, shouldStop, echoStartText = true) {
    console.log(`✍️ [Generate] seed="${startText.slice(0, 20).replace(/\n/g, '\\n')}", len=${generateLen}, temp=${temperature}`);

    let tokens = tokenizer.encode(startText);
    if (tokens.length === 0) {
        const t = tokenizer.randomStartToken();
        tokens = [t];
        if (onChar && echoStartText) onChar(tokenizer.itos[t] || '');
    } else {
        if (echoStartText) {
            for (const c of startText) { if (onChar) onChar(c); }
        }
    }

    for (let i = 0; i < generateLen; i++) {
        if (shouldStop && shouldStop()) break;

        let inputTokens = tokens.slice(-contextLength);
        while (inputTokens.length < contextLength) inputTokens = [0, ...inputTokens];
        const posTokens = Array.from({ length: contextLength }, (_, j) => j);

        const xT = tf.tensor2d([inputTokens], [1, contextLength], 'int32');
        const pT = tf.tensor2d([posTokens], [1, contextLength], 'int32');

        const logitsFull = model.predict([xT, pT]);
        const lastLogits = tf.squeeze(tf.slice(logitsFull, [0, contextLength - 1, 0], [1, 1, -1]));
        const scaled = tf.div(lastLogits, Math.max(0.01, temperature));
        const probs = tf.softmax(scaled);
        const probsData = await probs.data();

        const nextToken = sampleFromProbs(probsData);
        tokens.push(nextToken);
        if (onChar) onChar(tokenizer.itos[nextToken] || '');

        tf.dispose([xT, pT, logitsFull, lastLogits, scaled, probs]);
        if (i % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    console.log(`✍️ [Generate] Done`);
}

function sampleFromProbs(probs) {
    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
        cumsum += probs[i];
        if (r <= cumsum) return i;
    }
    return probs.length - 1;
}

window.buildTransformerModel = buildTransformerModel;
window.generateText = generateText;
window.CausalSelfAttentionLayer = CausalSelfAttentionLayer;
