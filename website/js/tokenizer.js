// tokenizer.js — Character-level tokenizer (matches character_gpt.ipynb logic)
// stoi: char -> int (1-indexed), pad token = 0, maps to '~'
// itos: int -> char

class CharTokenizer {
    constructor() {
        this.stoi = {};
        this.itos = {};
        this.vocabSize = 0;
        console.log('📝 [Tokenizer] CharTokenizer instance created');
    }

    /**
     * Build vocabulary from raw text
     * @param {string} text 
     */
    build(text) {
        const chars = Array.from(new Set(text.split(''))).sort();
        this.stoi = {};
        this.itos = {};

        // pad token = 0 → '~'
        this.stoi['~'] = 0;
        this.itos[0] = '~';

        chars.forEach((c, i) => {
            if (c !== '~') {
                this.stoi[c] = i + 1;
                this.itos[i + 1] = c;
            }
        });

        this.vocabSize = Object.keys(this.stoi).length;
        console.log(`📝 [Tokenizer] Vocabulary built — ${this.vocabSize} unique characters`);
        console.log(`📝 [Tokenizer] Sample chars:`, chars.slice(0, 20).join(''));
        return this.vocabSize;
    }

    /**
     * Encode string → array of ints
     * @param {string} text
     * @returns {number[]}
     */
    encode(text) {
        return text.split('').map(c => this.stoi[c] !== undefined ? this.stoi[c] : 0);
    }

    /**
     * Decode array of ints → string
     * @param {number[]} tokens
     * @returns {string}
     */
    decode(tokens) {
        return tokens.map(t => this.itos[t] || '').join('');
    }

    /**
     * Get a random valid start token index (avoids pad)
     */
    randomStartToken() {
        const keys = Object.keys(this.stoi).filter(k => k !== '~');
        const randomChar = keys[Math.floor(Math.random() * keys.length)];
        return this.stoi[randomChar];
    }
}

// Export as global
window.CharTokenizer = CharTokenizer;
