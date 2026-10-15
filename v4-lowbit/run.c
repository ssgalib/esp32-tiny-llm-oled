/* Inference for Llama-2 Transformer model in pure C (Flash Direct-Streaming for Non-PSRAM ESP32) */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifdef ESP32
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #include "esp_system.h"
    #define heap_alloc(size) malloc(size)
#else
    #define heap_alloc(size) malloc(size)
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim;            // transformer dimension
    int hidden_dim;     // for ffn layers
    int n_layers;       // number of layers
    int n_heads;        // number of query heads
    int n_kv_heads;     // number of key/value heads
    int vocab_size;     // vocabulary size
    int seq_len;        // max sequence length
} Config;

#define QB_MAGIC "QBRW"
#define QB_VERSION 1
#define QB_GROUP 32                      // weights per scale, fixed by tools/quantize_bits.py
#define QB_HEADER (16 + sizeof(Config))  // magic, version, bits, group, Config
#define MAX_LAYERS 16
#define MAX_DIM 4096                     // bounds the per-group x sums on the stack
#define RAM_HEADROOM 16384               // left free after copying weights (encode/generate allocate later)

// Where each weight block lives: memory-mapped flash, or a RAM copy.
typedef struct {
    int bits;
    const uint8_t* emb;    // block (n=dim, d=vocab); also the classifier
    const float* rms_att;  // [layers][dim]
    const float* rms_ffn;  // [layers][dim]
    const float* rms_final;
    const uint8_t *wq[MAX_LAYERS], *wk[MAX_LAYERS], *wv[MAX_LAYERS], *wo[MAX_LAYERS];
    const uint8_t *w1[MAX_LAYERS], *w2[MAX_LAYERS], *w3[MAX_LAYERS];
    void* ram[4 + 7 * MAX_LAYERS];  // RAM copies, freed with the transformer
    int n_ram;
} Weights;

typedef struct {
    float *x;           // activation at current time stamp
    float *xb;          // residual branch activation
    float *xb2;         // convenience buffer
    float *hb;          // hidden dimension buffer for ffn
    float *hb2;         // hidden dimension buffer for ffn
    float *q;           // query
    float *k;           // key
    float *v;           // value
    float *att;         // attention scores
    float *logits;      // output logits
    float* key_cache;   // KV cache
    float* value_cache; // KV cache
} RunState;

typedef struct {
    Config config;
    Weights w;
    RunState state;
} Transformer;

void free_run_state(RunState* s);

void malloc_run_state(RunState* s, Config* p) {
    if (p->n_heads == 0) {
        fprintf(stderr, "Error: n_heads is zero!\n");
        return;
    }
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    // Allocate runtime activation state in internal SRAM
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    
    // Safety Guard: Stop execution immediately if heap space is exhausted
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "\n❌ FATAL: Out of SRAM Heap memory! Try lowering max seq_len.\n");
        free_run_state(s);
        memset(s, 0, sizeof(RunState));
        return;
    }
}

void free_run_state(RunState* s) {
    if (s->x) free(s->x);
    if (s->xb) free(s->xb);
    if (s->xb2) free(s->xb2);
    if (s->hb) free(s->hb);
    if (s->hb2) free(s->hb2);
    if (s->q) free(s->q);
    if (s->att) free(s->att);
    if (s->logits) free(s->logits);
    if (s->key_cache) free(s->key_cache);
    if (s->value_cache) free(s->value_cache);
}

// A matrix block: float32 scales[d][groups], then codes[d][groups][4 * bits bytes].
static size_t qblock(int n, int d, int bits) {
    size_t groups = (n + QB_GROUP - 1) / QB_GROUP;
    return (size_t)d * groups * (sizeof(float) + QB_GROUP * bits / 8);
}

static void free_weights(Weights* w) {
    for (int i = 0; i < w->n_ram; i++) free(w->ram[i]);
    w->n_ram = 0;
}

// Point every weight block at the mapped model, then optionally copy as many blocks into RAM
// as fit while leaving RAM_HEADROOM free; the rest stay in flash. Every block is read once per
// token, so the share in RAM is what matters, not which blocks.
// Returns 0 if the model is too big for the flash region, otherwise the percent of weight
// bytes in RAM plus 1 (1 = all flash, 101 = all RAM).
static int place_weights(Transformer* t, const uint8_t* model, size_t model_size, int to_ram) {
    Config* p = &t->config;
    Weights* w = &t->w;
    int L = p->n_layers, dim = p->dim, hid = p->hidden_dim, bits = w->bits;
    int kv_dim = (dim * p->n_kv_heads) / p->n_heads;
    struct { const void** slot; size_t bytes; } sec[4 + 7 * MAX_LAYERS];
    int n = 0;
#define SECTION(field, size) (sec[n].slot = (const void**)&(field), sec[n++].bytes = (size))
    // Same order as tools/quantize_bits.py writes them.
    SECTION(w->emb, qblock(dim, p->vocab_size, bits));
    SECTION(w->rms_att, (size_t)L * dim * sizeof(float));
    for (int l = 0; l < L; l++) SECTION(w->wq[l], qblock(dim, dim, bits));
    for (int l = 0; l < L; l++) SECTION(w->wk[l], qblock(dim, kv_dim, bits));
    for (int l = 0; l < L; l++) SECTION(w->wv[l], qblock(dim, kv_dim, bits));
    for (int l = 0; l < L; l++) SECTION(w->wo[l], qblock(dim, dim, bits));
    SECTION(w->rms_ffn, (size_t)L * dim * sizeof(float));
    for (int l = 0; l < L; l++) SECTION(w->w1[l], qblock(dim, hid, bits));
    for (int l = 0; l < L; l++) SECTION(w->w2[l], qblock(hid, dim, bits));
    for (int l = 0; l < L; l++) SECTION(w->w3[l], qblock(dim, hid, bits));
    SECTION(w->rms_final, (size_t)dim * sizeof(float));
#undef SECTION

    size_t off = QB_HEADER;
    for (int i = 0; i < n; i++) {
        *sec[i].slot = model + off;
        off += sec[i].bytes;
    }
    if (off > model_size) {
        fprintf(stderr, "Model is larger than the flash region!\n");
        return 0;
    }
    size_t total = off - QB_HEADER, in_ram = 0;
    for (int i = 0; to_ram && i < n; i++) {
#ifdef ESP32
        if (esp_get_free_heap_size() < sec[i].bytes + RAM_HEADROOM) continue;
#endif
        void* copy = malloc(sec[i].bytes);
        if (!copy) continue;  // no block that large left; a smaller one may still fit
        memcpy(copy, *sec[i].slot, sec[i].bytes);
        *sec[i].slot = copy;
        w->ram[w->n_ram++] = copy;
        in_ram += sec[i].bytes;
    }
    return 1 + (int)(in_ram * 100 / total);
}

// The model comes from raw flash, so reject a header that can't be a real model
// (e.g. the partition still holds LittleFS) before sizing any allocation from it.
static int config_is_sane(const Config* c) {
    return c->dim > 0 && c->dim <= MAX_DIM && c->hidden_dim > 0 && c->hidden_dim <= MAX_DIM &&
           c->n_layers > 0 && c->n_layers <= MAX_LAYERS && c->n_heads > 0 && c->n_heads <= c->dim &&
           c->dim % c->n_heads == 0 && c->n_kv_heads > 0 && c->n_heads % c->n_kv_heads == 0 &&
           c->vocab_size > 0 && c->vocab_size <= 65536 && c->seq_len > 0;
}

// Returns 0 on a bad model. Weights start out in flash; see place_weights() to move some to RAM.
int build_transformer(Transformer *t, const uint8_t* model, size_t model_size) {
    memset(t, 0, sizeof(Transformer));
    int header[3];  // version, bits, group
    memcpy(header, model + 4, sizeof header);
    if (memcmp(model, QB_MAGIC, 4) != 0 || header[0] != QB_VERSION || header[2] != QB_GROUP ||
        !(header[1] == 1 || header[1] == 2 || header[1] == 4 || header[1] == 6 || header[1] == 8)) {
        fprintf(stderr, "Not a QBRW v1 model (run tools/quantize_bits.py)!\n");
        return 0;
    }
    t->w.bits = header[1];
    memcpy(&t->config, model + 16, sizeof(Config));
    if (!config_is_sane(&t->config)) {
        fprintf(stderr, "Invalid model header configuration!\n");
        memset(&t->config, 0, sizeof(Config));
        return 0;
    }

    // --- CAP SEQUENCE LENGTH FOR NON-PSRAM ESP32 SRAM LIMITS ---
    if (t->config.seq_len > 64) {
        t->config.seq_len = 64; 
    }

    malloc_run_state(&t->state, &t->config);
    if (!t->state.x) return 0;
    if (!place_weights(t, model, model_size, 0)) {
        memset(&t->config, 0, sizeof(Config));
        return 0;
    }
    return 1;
}

void free_transformer(Transformer* t) {
    free_weights(&t->w);
    free_run_state(&t->state);
}

// Unpack one group of QB_GROUP codes from a little-endian bitstream of 4 * bits bytes.
static inline void unpack_group(const uint8_t* src, int bits, uint8_t* q) {
    switch (bits) {
    case 8:
        memcpy(q, src, QB_GROUP);
        break;
    case 6:
        for (int i = 0; i < QB_GROUP; i += 4, src += 3) {
            q[i] = src[0] & 63;
            q[i + 1] = (src[0] >> 6) | ((src[1] & 15) << 2);
            q[i + 2] = (src[1] >> 4) | ((src[2] & 3) << 4);
            q[i + 3] = src[2] >> 2;
        }
        break;
    case 4:
        for (int i = 0; i < QB_GROUP; i += 2, src++) {
            q[i] = *src & 15;
            q[i + 1] = *src >> 4;
        }
        break;
    case 2:
        for (int i = 0; i < QB_GROUP; i += 4, src++) {
            q[i] = *src & 3;
            q[i + 1] = (*src >> 2) & 3;
            q[i + 2] = (*src >> 4) & 3;
            q[i + 3] = *src >> 6;
        }
        break;
    case 1:
        for (int i = 0; i < QB_GROUP; i += 8, src++) {
            for (int k = 0; k < 8; k++) q[i + k] = (*src >> k) & 1;
        }
        break;
    }
}

// Code q stands for (q - (2^bits - 1) / 2) * scale, so per group:
// sum((q - half) * x) * scale = (q . x - half * sum(x)) * scale, with sum(x) shared by all rows.
void matmul_q(const uint8_t* block, float* xout, const float* x, int n, int d, int bits) {
    int groups = (n + QB_GROUP - 1) / QB_GROUP;
    int group_bytes = QB_GROUP * bits / 8;
    const float* scales = (const float*)block;
    const uint8_t* codes = block + (size_t)d * groups * sizeof(float);
    float half = ((1 << bits) - 1) * 0.5f;

    float xsum[MAX_DIM / QB_GROUP];
    for (int g = 0; g < groups; g++) {
        int end = (g + 1) * QB_GROUP < n ? (g + 1) * QB_GROUP : n;
        xsum[g] = 0.0f;
        for (int j = g * QB_GROUP; j < end; j++) xsum[g] += x[j];
    }

    uint8_t q[QB_GROUP];
    for (int i = 0; i < d; i++) {
        const uint8_t* row = codes + (size_t)i * groups * group_bytes;
        const float* s = scales + (size_t)i * groups;
        float acc = 0.0f;
        for (int g = 0; g < groups; g++) {
            unpack_group(row + g * group_bytes, bits, q);
            const float* xg = x + g * QB_GROUP;
            int count = n - g * QB_GROUP < QB_GROUP ? n - g * QB_GROUP : QB_GROUP;
            float dot = 0.0f;
            for (int k = 0; k < count; k++) dot += q[k] * xg[k];
            acc += (dot - half * xsum[g]) * s[g];
        }
        xout[i] = acc;
    }
}

// x = dequantized row `token` of the embedding block.
static void embed(const uint8_t* block, float* x, int token, int dim, int vocab, int bits) {
    int groups = (dim + QB_GROUP - 1) / QB_GROUP;
    int group_bytes = QB_GROUP * bits / 8;
    const float* s = (const float*)block + (size_t)token * groups;
    const uint8_t* row = block + (size_t)vocab * groups * sizeof(float) + (size_t)token * groups * group_bytes;
    float half = ((1 << bits) - 1) * 0.5f;
    uint8_t q[QB_GROUP];
    for (int g = 0; g < groups; g++) {
        unpack_group(row + g * group_bytes, bits, q);
        for (int k = 0; k < QB_GROUP && g * QB_GROUP + k < dim; k++) {
            x[g * QB_GROUP + k] = (q[k] - half) * s[g];
        }
    }
}

void rmsnorm(const float* weight, float* o, float* x, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    Weights* w = &transformer->w;
    RunState* s = &transformer->state;
    int bits = w->bits;
    
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    embed(w->emb, x, token, dim, p->vocab_size, bits);

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm(w->rms_att + l * dim, s->xb, x, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        matmul_q(w->wq[l], s->q, s->xb, dim, dim, bits);
        matmul_q(w->wk[l], s->k, s->xb, dim, kv_dim, bits);
        matmul_q(w->wv[l], s->v, s->xb, dim, kv_dim, bits);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        matmul_q(w->wo[l], s->xb2, s->xb, dim, dim, bits);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm(w->rms_ffn + l * dim, s->xb, x, dim);

        matmul_q(w->w1[l], s->hb, s->xb, dim, hidden_dim, bits);
        matmul_q(w->w3[l], s->hb2, s->xb, dim, hidden_dim, bits);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul_q(w->w2[l], s->xb, s->hb, hidden_dim, dim, bits);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm(w->rms_final, x, x, dim);
    matmul_q(w->emb, s->logits, x, p->dim, p->vocab_size, bits);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer implementation

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int build_tokenizer(Tokenizer* t, const uint8_t* data, size_t size, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)calloc(vocab_size, sizeof(char*));  // zeroed so a partial load frees cleanly
    t->vocab_scores = (float*)heap_alloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    if (!t->vocab || !t->vocab_scores) {
        fprintf(stderr, "Failed to allocate memory for tokenizer structures.\n");
        return 0;
    }

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = fmemopen((void*)data, size, "rb");
    if (!file) { 
        fprintf(stderr, "couldn't open tokenizer data\n"); 
        return 0; 
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1 ||
        t->max_token_length == 0 || t->max_token_length > 256) { 
        fprintf(stderr, "failed tokenizer read\n"); 
        fclose(file); 
        return 0; 
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1 ||
            fread(&len, sizeof(int), 1, file) != 1 ||
            len <= 0 || len > (int)t->max_token_length ||
            !(t->vocab[i] = (char *)heap_alloc(len + 1)) ||
            fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "Invalid tokenizer data at entry %d\n", i);
            fclose(file);
            return 0;
        }
        t->vocab[i][len] = '\0';
    }
    fclose(file);
    return 1;
}

void free_tokenizer(Tokenizer* t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) { 
            if (t->vocab[i]) free(t->vocab[i]); 
        }
        free(t->vocab);
    }
    if (t->vocab_scores) free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void on_token(const char *piece);  // implemented in llama_oled.ino

void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
    on_token(piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); return; }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler implementation

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    if (sampler->probindex) free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Generation utility functions

long time_in_ms() {
#ifdef ESP32
    return (long)(xTaskGetTickCount() * portTICK_PERIOD_MS);
#else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    if (transformer->config.dim == 0 || transformer->config.n_heads == 0 || !transformer->state.x) {
        fprintf(stderr, "Cannot generate, invalid Transformer allocation.\n");
        return;
    }

    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Expected at least 1 prompt token\n");
        free(prompt_tokens);
        return;
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    // Do not exceed configured context sequence window
    if (steps > transformer->config.seq_len) {
        steps = transformer->config.seq_len;
    }

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }

#ifdef ESP32
        vTaskDelay(1);
#endif
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

int llama_ram_percent = 0;  // share of the running story's weight bytes held in RAM
int llama_bits = 0;

// Returns 0 if the model or tokenizer in flash is missing or invalid, otherwise 1.
int run_llama(const uint8_t* model, size_t model_size, const uint8_t* tokenizer_data, size_t tokenizer_size,
              float temperature, float topp, int steps, const char* prompt, int weights_in_ram) {
    Transformer transformer;
    if (!build_transformer(&transformer, model, model_size)) {
        printf("Failed to load model architecture or allocate state memory.\n");
        free_transformer(&transformer);
        return 0;
    }

    Tokenizer tokenizer;
    if (!build_tokenizer(&tokenizer, tokenizer_data, tokenizer_size, transformer.config.vocab_size)) {
        free_tokenizer(&tokenizer);
        free_transformer(&transformer);
        return 0;
    }

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, (unsigned long long)time(NULL));

    // Last, so the RAM copy only uses memory nothing else needs.
    int placed = place_weights(&transformer, model, model_size, weights_in_ram);
    llama_ram_percent = placed - 1;
    llama_bits = transformer.w.bits;

    generate(&transformer, &tokenizer, &sampler, (char*)prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 1;
}
