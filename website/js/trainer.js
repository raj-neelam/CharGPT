// trainer.js — Training loop with LR scheduling, WMA loss, and stop support

class Trainer {
    constructor() {
        this.model = null;
        this.optimizer = null;
        this.stopRequested = false;
        this.isTraining = false;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.lossHistory = [];
        this.wmaLossHistory = [];
        this.epochLosses = [];
        this.WMA_WINDOW = 50;

        this.onBatchEnd = null;
        this.onEpochEnd = null;
        this.onTrainingEnd = null;
        this.onEpochStart = null;

        console.log('🔄 [Train] Trainer instance created');
    }

    getLR(epoch, totalEpochs, lrStart, lrEnd) {
        if (totalEpochs <= 1) return lrEnd;
        return lrStart + (lrEnd - lrStart) * (epoch / (totalEpochs - 1));
    }

    computeWMA() {
        const window = this.lossHistory.slice(-this.WMA_WINDOW);
        if (window.length === 0) return 0;
        let totalWeight = 0, weightedSum = 0;
        for (let i = 0; i < window.length; i++) {
            const w = i + 1;
            weightedSum += window[i] * w;
            totalWeight += w;
        }
        return weightedSum / totalWeight;
    }

    /**
     * Create batches from encoded token array.
     */
    createBatches(encodedData, contextLength, batchSize, trainFraction = 0.95) {
        const splitIdx = Math.floor(encodedData.length * trainFraction);
        const trainData = encodedData.slice(0, splitIdx);
        const testData = encodedData.slice(splitIdx);
        const stride = Math.max(1, Math.floor(contextLength / 2));

        const makeSamples = (data) => {
            const samples = [];
            for (let start = 0; start + contextLength + 1 <= data.length; start += stride) {
                samples.push({
                    X: data.slice(start, start + contextLength),
                    y: data.slice(start + 1, start + contextLength + 1)
                });
            }
            return samples;
        };

        const shuffleArray = (arr) => {
            for (let i = arr.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [arr[i], arr[j]] = [arr[j], arr[i]];
            }
            return arr;
        };

        const chunk = (arr, size) =>
            Array.from({ length: Math.ceil(arr.length / size) }, (_, i) =>
                arr.slice(i * size, i * size + size)
            ).filter(b => b.length === size);

        const trainBatches = chunk(shuffleArray(makeSamples(trainData)), batchSize);
        const testBatches = chunk(makeSamples(testData), batchSize);

        console.log(`📦 [Train] Batches: ${trainBatches.length} train, ${testBatches.length} test (stride=${stride}, contextLen=${contextLength})`);
        return { trainBatches, testBatches };
    }

    /**
     * Convert a batch of {X, y} samples to flat Int32Arrays
     */
    batchToArrays(batch) {
        const batchSize = batch.length;
        const contextLength = batch[0].X.length;
        const XData = new Int32Array(batchSize * contextLength);
        const yData = new Int32Array(batchSize * contextLength);
        for (let b = 0; b < batchSize; b++) {
            for (let t = 0; t < contextLength; t++) {
                XData[b * contextLength + t] = batch[b].X[t];
                yData[b * contextLength + t] = batch[b].y[t];
            }
        }
        return { XData, yData, batchSize, contextLength };
    }

    /**
     * Main training function
     */
    async train(model, encodedData, config) {
        const tf = window.tf;
        const {
            contextLength, batchSize, epochs, lrStart, lrEnd, vocabSize,
            onBatchEnd, onEpochEnd, onTrainingEnd, onEpochStart, onUntrainedSample
        } = config;

        this.model = model;
        this.stopRequested = false;
        this.isTraining = true;
        this.totalEpochs = epochs;
        this.lossHistory = [];
        this.wmaLossHistory = [];
        this.epochLosses = [];

        console.log(`🔄 [Train] Starting: epochs=${epochs}, batchSize=${batchSize}, ctx=${contextLength}, lrStart=${lrStart}, lrEnd=${lrEnd}`);

        const { trainBatches, testBatches } = this.createBatches(encodedData, contextLength, batchSize);

        if (trainBatches.length === 0) {
            console.warn('⚠️ [Warn] No training batches! Data too small for context length?');
            this.isTraining = false;
            return;
        }

        // Build Adam optimizer with initial LR
        const initialLR = this.getLR(0, epochs, lrStart, lrEnd);
        this.optimizer = tf.train.adam(initialLR);

        // Generate untrained sample (epoch 0)
        if (onUntrainedSample) {
            console.log(`✍️ [Generate] Epoch 0 (untrained) sample starting`);
            await onUntrainedSample();
        }

        // Training epochs
        for (let epoch = 0; epoch < epochs; epoch++) {
            if (this.stopRequested) {
                console.log(`🛑 [Train] Stopped before epoch ${epoch + 1}`);
                break;
            }

            this.currentEpoch = epoch;
            const lr = this.getLR(epoch, epochs, lrStart, lrEnd);

            // Update optimizer learning rate
            this.optimizer.learningRate = lr;

            if (onEpochStart) onEpochStart(epoch, lr);
            console.log(`🔄 [Train] Epoch ${epoch + 1}/${epochs} — LR: ${lr.toExponential(3)}`);

            // Shuffle batches each epoch
            const shuffled = [...trainBatches].sort(() => Math.random() - 0.5);

            // Batch loop
            for (let bi = 0; bi < shuffled.length; bi++) {
                if (this.stopRequested) break;

                const lossVal = await this._trainStep(shuffled[bi], vocabSize, contextLength, batchSize);

                if (isNaN(lossVal)) {
                    console.warn(`⚠️ [Train] NaN loss at batch ${bi}, skipping`);
                    continue;
                }

                this.lossHistory.push(lossVal);
                const wma = this.computeWMA();
                this.wmaLossHistory.push(wma);

                if (onBatchEnd) onBatchEnd(bi, shuffled.length, wma, lr, lossVal);

                // Yield to browser UI
                await new Promise(r => setTimeout(r, 0));
            }

            if (this.stopRequested) break;

            // Test evaluation
            const avgTestLoss = await this._evaluateTest(testBatches, vocabSize, contextLength, batchSize);
            this.epochLosses.push(avgTestLoss);

            console.log(`📊 [Loss] Epoch ${epoch + 1} — Avg Test Loss: ${avgTestLoss.toFixed(4)}`);
            if (onEpochEnd) await onEpochEnd(epoch, avgTestLoss);
        }

        this.isTraining = false;
        console.log(`🔄 [Train] Done. Epochs completed: ${Math.min(this.currentEpoch + 1, epochs)}`);
        if (onTrainingEnd) onTrainingEnd();
    }

    /**
     * Single training step using optimizer.minimize()
     */
    async _trainStep(batch, vocabSize, contextLength, batchSize) {
        const tf = window.tf;
        const { XData, yData } = this.batchToArrays(batch);
        const actualBatch = batch.length;

        const xTensor = tf.tensor2d(XData, [actualBatch, contextLength], 'int32');
        const yTensor = tf.tensor2d(yData, [actualBatch, contextLength], 'int32');

        // Position indices: [0, 1, ..., contextLength-1] repeated for each sample in batch
        const posData = new Int32Array(actualBatch * contextLength);
        for (let b = 0; b < actualBatch; b++) {
            for (let t = 0; t < contextLength; t++) {
                posData[b * contextLength + t] = t;
            }
        }
        const posTensor = tf.tensor2d(posData, [actualBatch, contextLength], 'int32');

        let lossVal = 0;

        const lossTensor = this.optimizer.minimize(() => {
            const logits = this.model.apply([xTensor, posTensor], { training: true });
            const logitsFlat = tf.reshape(logits, [-1, vocabSize]);
            const yFlat = tf.reshape(yTensor, [-1]);
            const yOneHot = tf.oneHot(yFlat, vocabSize);
            const logProbs = tf.logSoftmax(logitsFlat, -1);
            const loss = tf.neg(tf.mean(tf.sum(tf.mul(yOneHot, logProbs), -1)));
            tf.dispose([logits, logitsFlat, yFlat, yOneHot, logProbs]);
            return loss;
        }, true);  // null varList → TF.js auto-collects all model variables via gradient tape

        if (lossTensor) {
            const data = await lossTensor.data();
            lossVal = data[0];
            lossTensor.dispose();
        }

        tf.dispose([xTensor, yTensor, posTensor]);
        return lossVal;
    }

    /**
     * Evaluate on test set (no gradients)
     */
    async _evaluateTest(testBatches, vocabSize, contextLength, batchSize) {
        const tf = window.tf;
        if (testBatches.length === 0) {
            // No test split — return last WMA train loss
            const lastWma = this.wmaLossHistory.at(-1);
            return (lastWma !== undefined && !isNaN(lastWma)) ? lastWma : 0;
        }

        let totalLoss = 0, count = 0;

        for (const batch of testBatches) {
            if (this.stopRequested) break;
            const { XData, yData } = this.batchToArrays(batch);
            const actualBatch = batch.length;

            const posData = new Int32Array(actualBatch * contextLength);
            for (let b = 0; b < actualBatch; b++)
                for (let t = 0; t < contextLength; t++)
                    posData[b * contextLength + t] = t;

            const xTensor = tf.tensor2d(XData, [actualBatch, contextLength], 'int32');
            const yTensor = tf.tensor2d(yData, [actualBatch, contextLength], 'int32');
            const posTensor = tf.tensor2d(posData, [actualBatch, contextLength], 'int32');

            const logits = this.model.predict([xTensor, posTensor]);
            const logitsFlat = tf.reshape(logits, [-1, vocabSize]);
            const yFlat = tf.reshape(yTensor, [-1]);
            const yOneHot = tf.oneHot(yFlat, vocabSize);
            const logProbs = tf.logSoftmax(logitsFlat, -1);
            const loss = tf.neg(tf.mean(tf.sum(tf.mul(yOneHot, logProbs), -1)));

            const lossVal = (await loss.data())[0];
            if (!isNaN(lossVal)) { totalLoss += lossVal; count++; }

            tf.dispose([xTensor, yTensor, posTensor, logits, logitsFlat, yFlat, yOneHot, logProbs, loss]);
        }

        return count > 0 ? totalLoss / count : 0;
    }

    stop() {
        this.stopRequested = true;
        console.log(`🛑 [Train] Stop requested`);
    }

    reset() {
        this.stopRequested = false;
        this.isTraining = false;
        this.lossHistory = [];
        this.wmaLossHistory = [];
        this.epochLosses = [];
        this.currentEpoch = 0;
        if (this.optimizer) {
            try { this.optimizer.dispose(); } catch (e) { }
            this.optimizer = null;
        }
        console.log(`🔄 [Train] Trainer reset`);
    }
}

window.Trainer = Trainer;
