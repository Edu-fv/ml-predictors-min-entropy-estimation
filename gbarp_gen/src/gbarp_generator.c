/*
 * gbAR(p) Generator
 * Copyright (c) 2026 Javier Blanco-Romero
 *
 * X_t = (Σ a+_i * X_{t-i} + Σ a-_i * (1-X_{t-i}) + b * e_t) mod 2
 *
 * All randomness (categorical + noise) is pre-read before the generation loop.
 * Secure mode (default): /dev/urandom. Fast mode: xorshift64 seeded from /dev/urandom.
 * Categorical sampling uses 4 bytes per step (big-endian uint32 / 0xFFFFFFFF).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

/* Randomness sources */

static int read_urandom(uint8_t *buffer, size_t len)
{
    FILE *fp = fopen("/dev/urandom", "rb");
    if (!fp)
        return -1;
    size_t nread = fread(buffer, 1, len, fp);
    fclose(fp);
    return (nread == len) ? 0 : -1;
}

static uint64_t prng_state = 0;

static void prng_seed(uint64_t seed)
{
    prng_state = seed ? seed : 0x123456789ABCDEF0ULL;
}

static uint64_t prng_next(void)
{
    uint64_t x = prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    prng_state = x;
    return x;
}

static int fill_random(uint8_t *buffer, size_t len, int use_urandom)
{
    if (use_urandom)
        return read_urandom(buffer, len);
    for (size_t i = 0; i < len; i++)
        buffer[i] = (uint8_t)(prng_next() & 0xFF);
    return 0;
}

/* Core generator */

int gbarp_generate(const double *alpha, int p, double beta,
                   uint8_t *output, size_t num_bytes,
                   size_t burn_in_bytes, int use_urandom)
{
    if (!alpha || !output || p < 1 || num_bytes == 0)
        return -1;

    double *P_abs = (double *)malloc((p + 1) * sizeof(double));
    if (!P_abs)
        return -2;

    double sum = 0.0;
    for (int i = 0; i < p; i++) {
        P_abs[i] = fabs(alpha[i]);
        sum += P_abs[i];
    }
    P_abs[p] = beta;
    sum += beta;
    for (int i = 0; i <= p; i++)
        P_abs[i] /= sum;

    int *alpha_positive = (int *)malloc(p * sizeof(int));
    if (!alpha_positive) {
        free(P_abs);
        return -2;
    }
    for (int i = 0; i < p; i++)
        alpha_positive[i] = (alpha[i] >= 0) ? 1 : 0;

    uint8_t *history = (uint8_t *)calloc(p, sizeof(uint8_t));
    if (!history) {
        free(P_abs);
        free(alpha_positive);
        return -2;
    }

    size_t num_bits   = num_bytes * 8;
    size_t burn_in    = burn_in_bytes * 8;
    size_t total_bits = num_bits + burn_in;
    size_t cat_size   = total_bits * 4;
    size_t noise_size = (total_bits + 7) / 8;

    uint8_t *cat_buf = (uint8_t *)malloc(cat_size);
    uint8_t *noise   = (uint8_t *)malloc(noise_size);
    if (!cat_buf || !noise) {
        free(P_abs); free(alpha_positive); free(history);
        free(cat_buf); free(noise);
        return -2;
    }

    if (!use_urandom) {
        uint64_t seed;
        if (read_urandom((uint8_t *)&seed, sizeof(seed)) != 0) {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            seed = (uint64_t)tv.tv_sec ^ (uint64_t)tv.tv_usec;
        }
        prng_seed(seed);
    }

    if (fill_random(cat_buf, cat_size, use_urandom) != 0 ||
        fill_random(noise, noise_size, use_urandom) != 0) {
        free(P_abs); free(alpha_positive); free(history);
        free(cat_buf); free(noise);
        return -3;
    }

    memset(output, 0, num_bytes);
    int hist_idx = 0;
    size_t output_bit = 0;

    for (size_t t = 0; t < total_bits; t++) {
        size_t co = t * 4;
        uint32_t raw = ((uint32_t)cat_buf[co] << 24)
                     | ((uint32_t)cat_buf[co + 1] << 16)
                     | ((uint32_t)cat_buf[co + 2] << 8)
                     |  (uint32_t)cat_buf[co + 3];
        double r = (double)raw / (double)0xFFFFFFFFU;

        int k = p;
        double cumsum = 0.0;
        for (int i = 0; i <= p; i++) {
            cumsum += P_abs[i];
            if (r <= cumsum) { k = i; break; }
        }

        uint8_t X_t;
        if (k == p) {
            X_t = (noise[t / 8] >> (7 - (t % 8))) & 1;
        } else {
            int hist_pos = (hist_idx - (k + 1) + p) % p;
            X_t = alpha_positive[k] ? history[hist_pos] : (1 - history[hist_pos]);
        }

        history[hist_idx] = X_t;
        hist_idx = (hist_idx + 1) % p;

        if (t >= burn_in) {
            output[output_bit / 8] |= (X_t << (7 - (output_bit % 8)));
            output_bit++;
        }
    }

    free(P_abs); free(alpha_positive); free(history);
    free(cat_buf); free(noise);
    return 0;
}

const char *gbarp_version(void) { return "1.0.0"; }


#ifdef STANDALONE

static double get_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void print_usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <output_file> <num_bytes> -a <v0,v1,...> [options]\n\n"
        "Options:\n"
        "  -a <v0,v1,...>     Coefficient vector (required). Negatives flip.\n"
        "  -p <int>           Validates vector length (optional)\n"
        "  --burn-in <int>    Burn-in bytes (default: 10000)\n"
        "  --fast             xorshift64 PRNG instead of /dev/urandom\n",
        prog);
}

static int parse_alpha_vector(const char *str, double **out_alpha, int *out_p)
{
    int count = 1;
    for (const char *c = str; *c; c++)
        if (*c == ',') count++;

    double *alpha = (double *)malloc(count * sizeof(double));
    if (!alpha) return -1;

    char *buf = strdup(str);
    if (!buf) { free(alpha); return -1; }

    char *tok = strtok(buf, ",");
    for (int i = 0; i < count && tok; i++, tok = strtok(NULL, ","))
        alpha[i] = atof(tok);

    free(buf);
    *out_alpha = alpha;
    *out_p = count;
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3) { print_usage(argv[0]); return 1; }

    const char *output_file = argv[1];
    size_t num_bytes = (size_t)atoll(argv[2]);
    int p_given = -1;
    const char *alpha_str = NULL;
    size_t burn_in_bytes = 10000;
    int use_urandom = 1;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            p_given = atoi(argv[++i]);
            if (p_given < 1 || p_given > 64) {
                fprintf(stderr, "Error: p must be in [1, 64]\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-a") == 0 && i + 1 < argc) {
            alpha_str = argv[++i];
        } else if (strcmp(argv[i], "--burn-in") == 0 && i + 1 < argc) {
            burn_in_bytes = (size_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "--fast") == 0) {
            use_urandom = 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!alpha_str) {
        fprintf(stderr, "Error: -a is required\n");
        print_usage(argv[0]);
        return 1;
    }

    double *alpha = NULL;
    int p;
    if (parse_alpha_vector(alpha_str, &alpha, &p) != 0) {
        fprintf(stderr, "Failed to parse alpha vector\n");
        return 1;
    }
    if (p_given >= 0 && p != p_given) {
        fprintf(stderr, "Error: vector length %d != p=%d\n", p, p_given);
        free(alpha);
        return 1;
    }

    double sum_abs = 0.0;
    for (int i = 0; i < p; i++)
        sum_abs += fabs(alpha[i]);
    if (sum_abs >= 1.0) {
        fprintf(stderr, "Error: sum(|alpha|) = %.4f >= 1\n", sum_abs);
        free(alpha);
        return 1;
    }
    double beta = 1.0 - sum_abs;

    printf("p=%d  beta=%.4f  burn_in=%zu  source=%s\n",
           p, beta, burn_in_bytes,
           use_urandom ? "/dev/urandom" : "xorshift64");
    printf("alpha=[");
    for (int i = 0; i < p; i++)
        printf("%s%.4f", i ? "," : "", alpha[i]);
    printf("]\n");

    uint8_t *output = (uint8_t *)malloc(num_bytes);
    if (!output) {
        fprintf(stderr, "Cannot allocate %zu bytes\n", num_bytes);
        free(alpha);
        return 1;
    }

    double start = get_time_ms();
    int ret = gbarp_generate(alpha, p, beta, output, num_bytes, burn_in_bytes, use_urandom);
    double elapsed = get_time_ms() - start;

    if (ret != 0) {
        fprintf(stderr, "Generation failed (code %d)\n", ret);
        free(alpha); free(output);
        return 1;
    }

    printf("%.2f ms  %.2f MB/s\n",
           elapsed, (num_bytes / (1024.0 * 1024.0)) / (elapsed / 1000.0));

    FILE *fp = fopen(output_file, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open %s\n", output_file);
        free(alpha); free(output);
        return 1;
    }
    size_t written = fwrite(output, 1, num_bytes, fp);
    fclose(fp);

    if (written != num_bytes) {
        fprintf(stderr, "Write failed\n");
        free(alpha); free(output);
        return 1;
    }

    printf("Wrote %zu bytes to %s\n", num_bytes, output_file);
    free(alpha); free(output);
    return 0;
}

#endif
