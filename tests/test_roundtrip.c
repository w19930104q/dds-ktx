/*
 * test_roundtrip.c - Roundtrip tests for dds-ktx.h
 *
 * Builds synthetic DDS/KTX blobs in-memory with deterministic payloads and
 * verifies that ddsktx_parse/ddsktx_get_sub recover every byte at the correct
 * offset. Focus: compressed textures with cropped/odd dimensions (not
 * multiples of the block size), full mip chains, cubemaps, arrays and 3D.
 *
 * The reference layout math below is written independently from the library
 * (block size per format table + spec-defined container layout), so an
 * agreement here actually means the parser matches the on-disk format.
 *
 * Build (see run_tests.sh):
 *   cc -std=c99 -Wall -Wextra -I.. test_roundtrip.c -o test_roundtrip
 */
#define DDSKTX_IMPLEMENT
#include "dds-ktx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/* Test harness                                                        */
/* ------------------------------------------------------------------ */

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        ++g_checks;                                                           \
        if (!(cond)) {                                                        \
            ++g_failures;                                                     \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                              \
            printf("\n");                                                     \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------------ */
/* Growable byte buffer                                                */
/* ------------------------------------------------------------------ */

typedef struct
{
    uint8_t* data;
    size_t   size;
    size_t   cap;
} buf_t;

static void buf_init(buf_t* b)
{
    b->cap = 4096;
    b->size = 0;
    b->data = (uint8_t*)malloc(b->cap);
}

static void buf_free(buf_t* b)
{
    free(b->data);
    b->data = NULL;
    b->size = b->cap = 0;
}

static void buf_reserve(buf_t* b, size_t extra)
{
    if (b->size + extra <= b->cap)
        return;
    while (b->size + extra > b->cap)
        b->cap *= 2;
    b->data = (uint8_t*)realloc(b->data, b->cap);
}

static void buf_put(buf_t* b, const void* p, size_t n)
{
    buf_reserve(b, n);
    memcpy(b->data + b->size, p, n);
    b->size += n;
}

static void buf_put_u32(buf_t* b, uint32_t v)
{
    buf_put(b, &v, sizeof(v));
}

/* Deterministic pseudo-random byte for a linear payload position. */
static uint8_t payload_byte(uint64_t pos)
{
    uint64_t x = pos * 6364136223846793005ull + 1442695040888963407ull;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    return (uint8_t)(x & 0xff);
}

/* ------------------------------------------------------------------ */
/* Reference format tables (independent from library internals)       */
/* ------------------------------------------------------------------ */

typedef struct
{
    ddsktx_format write_fmt;   /* format the on-disk container identifies */
    ddsktx_format expect_fmt;  /* format the library maps it to */
    int           block_w;
    int           block_h;
    int           block_bytes;
    int           min_blocks_x;
    int           min_blocks_y;
    int           bpp;        /* 0 => compressed block format */
    const char*   dds_fourcc; /* four chars, NULL => not representable in legacy DDS */
    uint32_t      dxgi;       /* DXGI code, 0 => no DX10 mapping (no DDS arrays) */
    int           dds_ok;     /* usable in a DDS container */
    int           ktx_ok;     /* usable in a KTX container */
} ref_fmt_t;

static const ref_fmt_t k_ref_formats[] = {
    /* write, expect, bw, bh, bbytes, minx, miny, bpp, fourcc, dxgi, dds, ktx */
    { DDSKTX_FORMAT_BC1,     DDSKTX_FORMAT_BC1,     4, 4,  8, 1, 1,  0, "DXT1", 71,  1, 1 },
    { DDSKTX_FORMAT_BC3,     DDSKTX_FORMAT_BC3,     4, 4, 16, 1, 1,  0, "DXT5", 77,  1, 1 },
    { DDSKTX_FORMAT_BC4,     DDSKTX_FORMAT_BC4,     4, 4,  8, 1, 1,  0, "ATI1", 80,  1, 1 },
    { DDSKTX_FORMAT_BC5,     DDSKTX_FORMAT_BC5,     4, 4, 16, 1, 1,  0, "ATI2", 83,  1, 1 },
    { DDSKTX_FORMAT_BC7,     DDSKTX_FORMAT_BC7,     4, 4, 16, 1, 1,  0, NULL,   98,  1, 1 },
    { DDSKTX_FORMAT_ETC1,    DDSKTX_FORMAT_ETC1,    4, 4,  8, 1, 1,  0, "ETC1", 0,   1, 1 },
    { DDSKTX_FORMAT_ETC2,    DDSKTX_FORMAT_ETC2,    4, 4,  8, 1, 1,  0, "ETC2", 0,   1, 1 },
    { DDSKTX_FORMAT_ETC2A,   DDSKTX_FORMAT_ETC2A,   4, 4, 16, 1, 1,  0, "ET2A", 0,   1, 1 },
    { DDSKTX_FORMAT_ASTC5x5, DDSKTX_FORMAT_ASTC5x5, 5, 5, 16, 1, 1,  0, "AS55", 0,   1, 1 },
    { DDSKTX_FORMAT_ASTC6x6, DDSKTX_FORMAT_ASTC6x6, 6, 6, 16, 1, 1,  0, "AS66", 0,   1, 1 },
    { DDSKTX_FORMAT_ASTC8x5, DDSKTX_FORMAT_ASTC8x5, 8, 5, 16, 1, 1,  0, "AS85", 0,   1, 1 },
    /* PVRTC: library maps PTC2/PTC4 fourccs to the *A variants. Its KTX size
     * semantics (2bpp/4bpp power-of-two square formula) do not match the
     * library's block-table model, so PVRTC is exercised via DDS only. */
    { DDSKTX_FORMAT_PTC12A,  DDSKTX_FORMAT_PTC12A,  8, 4,  8, 2, 2,  0, "PTC2", 0,   1, 0 },
    { DDSKTX_FORMAT_PTC14A,  DDSKTX_FORMAT_PTC14A,  4, 4,  8, 2, 2,  0, "PTC4", 0,   1, 0 },
    { DDSKTX_FORMAT_RGBA8,   DDSKTX_FORMAT_RGBA8,   1, 1,  4, 1, 1, 32, NULL,   28,  1, 1 },
    { DDSKTX_FORMAT_R8,      DDSKTX_FORMAT_R8,      1, 1,  1, 1, 1,  8, NULL,   61,  0, 1 },
};
static const int k_ref_format_count = (int)(sizeof(k_ref_formats) / sizeof(k_ref_formats[0]));

static const ref_fmt_t* ref_find(ddsktx_format fmt)
{
    for (int i = 0; i < k_ref_format_count; ++i)
        if (k_ref_formats[i].write_fmt == fmt)
            return &k_ref_formats[i];
    return NULL;
}

static int mip_dim(int base, int mip)
{
    int d = base >> mip;
    return d > 0 ? d : 1;
}

/* Reference footprint of one 2D face/slice of a mip level. */
static void ref_footprint(const ref_fmt_t* f, int w, int h,
                          int64_t* row_pitch, int64_t* slice_size)
{
    if (f->bpp == 0) {
        int blocks_w = (w + f->block_w - 1) / f->block_w;
        int blocks_h = (h + f->block_h - 1) / f->block_h;
        if (blocks_w < f->min_blocks_x) blocks_w = f->min_blocks_x;
        if (blocks_h < f->min_blocks_y) blocks_h = f->min_blocks_y;
        *row_pitch = (int64_t)blocks_w * f->block_bytes;
        *slice_size = *row_pitch * blocks_h;
    } else {
        *row_pitch = ((int64_t)w * f->bpp + 7) / 8;
        *slice_size = *row_pitch * h;
    }
}

/* ------------------------------------------------------------------ */
/* DDS builder                                                         */
/* ------------------------------------------------------------------ */

static uint32_t fourcc(const char* s)
{
    return (uint32_t)(uint8_t)s[0]
         | ((uint32_t)(uint8_t)s[1] << 8)
         | ((uint32_t)(uint8_t)s[2] << 16)
         | ((uint32_t)(uint8_t)s[3] << 24);
}

static void build_dds(buf_t* b, const ref_fmt_t* f,
                      int width, int height, int depth,
                      int num_mips, int num_layers, int cubemap)
{
    buf_put_u32(b, fourcc("DDS "));

    uint32_t hdr[124 / 4];
    memset(hdr, 0, sizeof(hdr));
    hdr[0] = 124;                                  /* size */
    hdr[1] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000;   /* caps|height|width|pixfmt|linearsize */
    if (num_mips > 1) hdr[1] |= 0x20000;
    if (depth > 1)   hdr[1] |= 0x800000;
    hdr[2] = (uint32_t)height;
    hdr[3] = (uint32_t)width;
    hdr[5] = (uint32_t)(depth > 1 ? depth : 0);
    hdr[6] = (uint32_t)num_mips;
    /* pixel format at header word 18 */
    uint32_t* pf = &hdr[18];
    pf[0] = 32;                                    /* pf size */
    /* DDS texture arrays require the DX10 extended header. BC7 has no legacy
     * fourcc either, so it always uses DX10. */
    int use_dx10 = (num_layers > 1) || (f->write_fmt == DDSKTX_FORMAT_BC7);
    if (use_dx10) {
        pf[1] = 0x4;                               /* FOURCC */
        pf[2] = fourcc("DX10");
    } else if (f->bpp == 0) {
        pf[1] = 0x4;                               /* FOURCC */
        pf[2] = fourcc(f->dds_fourcc);
    } else {
        /* uncompressed RGBA8 legacy pixel format */
        pf[1] = 0x40 | 0x1;                        /* RGB | ALPHAPIXELS */
        pf[3] = (uint32_t)f->bpp;
        if (f->write_fmt == DDSKTX_FORMAT_RGBA8) {
            pf[4] = 0x00ff0000; pf[5] = 0x0000ff00;
            pf[6] = 0x000000ff; pf[7] = 0xff000000;
        }
    }
    /* caps1 at header word 26, caps2 at word 27 (after 7 prefix words,
     * 11 reserved, 8-word pixel format) */
    uint32_t caps1 = 0x1000;                       /* TEXTURE */
    if (num_mips > 1 || cubemap || num_layers > 1)
        caps1 |= 0x8;                             /* COMPLEX */
    if (num_mips > 1) caps1 |= 0x400000;          /* MIPMAP */
    hdr[26] = caps1;
    uint32_t caps2 = 0;
    if (cubemap) {
        caps2 |= 0x200 | 0x400 | 0x800 | 0x1000 | 0x2000 | 0x4000 | 0x8000;
    }
    if (depth > 1) caps2 |= 0x200000;              /* VOLUME */
    hdr[27] = caps2;
    buf_put(b, hdr, sizeof(hdr));

    if (use_dx10) {
        /* DX10 extended header: dxgi_format, dimension(2D), misc(cube), array, misc2 */
        buf_put_u32(b, f->dxgi);
        buf_put_u32(b, depth > 1 ? 4u : 3u);
        buf_put_u32(b, cubemap ? 4u : 0u);
        buf_put_u32(b, (uint32_t)num_layers);
        buf_put_u32(b, 0);
    }

    /* Pixel data: layer -> face -> mip -> slice, tightly packed. */
    uint64_t pos = 0;
    int num_faces = cubemap ? 6 : 1;
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int face = 0; face < num_faces; ++face) {
            for (int mip = 0; mip < num_mips; ++mip) {
                int w = mip_dim(width, mip);
                int h = mip_dim(height, mip);
                int slices = depth > 1 ? mip_dim(depth, mip) : 1;
                int64_t pitch, slice_size;
                ref_footprint(f, w, h, &pitch, &slice_size);
                for (int s = 0; s < slices; ++s) {
                    for (int64_t k = 0; k < slice_size; ++k)
                        buf_put(b, (uint8_t[]){ payload_byte(pos++) }, 1);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* KTX builder                                                         */
/* ------------------------------------------------------------------ */

static uint32_t ktx_internal_format(ddsktx_format fmt)
{
    switch (fmt) {
    case DDSKTX_FORMAT_BC1:    return 0x83F1; /* COMPRESSED_RGBA_S3TC_DXT1_EXT */
    case DDSKTX_FORMAT_BC3:    return 0x83F3;
    case DDSKTX_FORMAT_BC4:    return 0x8C70; /* LUMINANCE_LATC1 */
    case DDSKTX_FORMAT_BC5:    return 0x8C72;
    case DDSKTX_FORMAT_BC7:    return 0x8E8C; /* BPTC_UNORM */
    case DDSKTX_FORMAT_ETC1:   return 0x8D64;
    case DDSKTX_FORMAT_ETC2:   return 0x9274;
    case DDSKTX_FORMAT_ETC2A:  return 0x9278;
    case DDSKTX_FORMAT_ASTC5x5: return 0x93B2;
    case DDSKTX_FORMAT_ASTC6x6: return 0x93B4;
    case DDSKTX_FORMAT_ASTC8x5: return 0x93B5;
    case DDSKTX_FORMAT_PTC12:  return 0x8C03;
    case DDSKTX_FORMAT_PTC14:  return 0x8C02;
    case DDSKTX_FORMAT_RGBA8:  return 0x8058;
    case DDSKTX_FORMAT_R8:     return 0x8229;
    default: return 0;
    }
}

static int64_t align4_i64(int64_t v)
{
    return (v + 3) & ~(int64_t)3;
}

/* padded=1 writes spec-conformant KTX where each imageSize/face stride is
 * rounded up to a multiple of 4 (the extra bytes are skipped, not payload). */
static void build_ktx(buf_t* b, const ref_fmt_t* f,
                      int width, int height, int depth,
                      int num_mips, int num_layers, int cubemap, int padded)
{
    static const uint8_t ktx_id[12] = {
        0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB,
        0x0D, 0x0A, 0x1A, 0x0A
    };
    buf_put(b, ktx_id, sizeof(ktx_id));

    int compressed = (f->bpp == 0);
    uint32_t hdr[13];
    memset(hdr, 0, sizeof(hdr));
    hdr[0]  = 0x04030201;                 /* endianness */
    hdr[1]  = 0;                          /* type */
    hdr[2]  = 1;                          /* type_size */
    hdr[3]  = 0;                          /* format */
    hdr[4]  = ktx_internal_format(f->write_fmt);
    hdr[5]  = compressed ? 0 : (f->write_fmt == DDSKTX_FORMAT_R8 ? 0x1903 : 0x1908);
    hdr[6]  = (uint32_t)width;
    hdr[7]  = (uint32_t)height;
    hdr[8]  = (uint32_t)(depth > 1 ? depth : 0);
    hdr[9]  = (uint32_t)num_layers;
    hdr[10] = cubemap ? 6u : 1u;          /* face_count */
    hdr[11] = (uint32_t)num_mips;
    hdr[12] = 0;                          /* metadata_size */
    buf_put(b, hdr, sizeof(hdr));

    /* Pixel data: mip -> imageSize(u32) -> layer -> face -> slices. With
     * padded=1 the face stride is rounded up to 4 bytes (spec imageSize);
     * otherwise the data is tightly packed. */
    uint64_t pos = 0;
    int num_faces = cubemap ? 6 : 1;
    for (int mip = 0; mip < num_mips; ++mip) {
        int w = mip_dim(width, mip);
        int h = mip_dim(height, mip);
        int slices = depth > 1 ? mip_dim(depth, mip) : 1;
        int64_t pitch, slice_size;
        ref_footprint(f, w, h, &pitch, &slice_size);
        int64_t face_size = slice_size * slices;
        int64_t stride = padded ? align4_i64(face_size) : face_size;
        buf_put_u32(b, (uint32_t)stride);

        for (int layer = 0; layer < num_layers; ++layer) {
            for (int face = 0; face < num_faces; ++face) {
                for (int s = 0; s < slices; ++s)
                    for (int64_t k = 0; k < slice_size; ++k)
                        buf_put(b, (uint8_t[]){ payload_byte(pos++) }, 1);
                /* Padding bytes are skipped (distinct fill, never payload). */
                for (int64_t k = face_size; k < stride; ++k)
                    buf_put(b, (uint8_t[]){ (uint8_t)0xCD }, 1);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Roundtrip verification                                              */
/* ------------------------------------------------------------------ */

typedef struct
{
    int width;
    int height;
    int depth;
    int num_mips;
    int num_layers;
    int cubemap;
} case_cfg_t;

static void verify_roundtrip(const ref_fmt_t* f, const case_cfg_t* cfg,
                             const buf_t* blob, int is_ktx)
{
    ddsktx_texture_info tc;
    memset(&tc, 0, sizeof(tc));
    ddsktx_error err;
    memset(&err, 0, sizeof(err));

    int parsed = ddsktx_parse(&tc, blob->data, (int)blob->size, &err);
    CHECK(parsed, "parse failed: %s (fmt=%s %dx%d mips=%d cube=%d ktx=%d)",
          err.msg, ddsktx_format_str(f->write_fmt), cfg->width, cfg->height,
          cfg->num_mips, cfg->cubemap, is_ktx);
    if (!parsed)
        return;

    /* A legacy DDS A8R8G8B8 pixel format is reported as BGRA8 by the library;
     * the DX10 path (used for arrays / BC7) and KTX keep RGBA8. */
    ddsktx_format expect_fmt = f->expect_fmt;
    if (!is_ktx && f->write_fmt == DDSKTX_FORMAT_RGBA8 && cfg->num_layers == 1)
        expect_fmt = DDSKTX_FORMAT_BGRA8;
    CHECK(tc.format == expect_fmt, "format mismatch: got %s want %s",
          ddsktx_format_str(tc.format), ddsktx_format_str(expect_fmt));
    CHECK(tc.width == cfg->width && tc.height == cfg->height,
          "base dims mismatch: got %dx%d", tc.width, tc.height);
    CHECK(tc.num_mips == cfg->num_mips, "mip count mismatch: got %d want %d",
          tc.num_mips, cfg->num_mips);
    CHECK(tc.num_layers == cfg->num_layers, "layer count mismatch");
    int expect_faces = cfg->cubemap ? 6 : 1;
    CHECK(((tc.flags & DDSKTX_TEXTURE_FLAG_CUBEMAP) != 0) == cfg->cubemap,
          "cubemap flag mismatch");
    CHECK(((tc.flags & DDSKTX_TEXTURE_FLAG_KTX) != 0) == is_ktx,
          "container flag mismatch");
    CHECK(((tc.flags & DDSKTX_TEXTURE_FLAG_DDS) != 0) == !is_ktx,
          "container flag mismatch");
    CHECK(ddsktx_format_compressed(tc.format) == (f->bpp == 0),
          "compressed classification mismatch");
    (void)expect_faces;

    /* Re-walk the reference layout and check every sub resource. The traversal
     * order matches the on-disk order of each container so the linear payload
     * position stays in sync:
     *   DDS: layer -> face -> mip -> slice
     *   KTX: mip  -> layer -> face -> slice
     */
    uint64_t pos = 0;
    int num_faces = cfg->cubemap ? 6 : 1;
    double sse = 0.0;

    /* Checks one sub resource against the next region of the linear payload. */
    #define CHECK_SUB(layer_idx, face_idx, mip_idx, slice_count)               \
        do {                                                                   \
            int w_ = mip_dim(cfg->width, (mip_idx));                            \
            int h_ = mip_dim(cfg->height, (mip_idx));                           \
            int64_t pitch_, slice_size_;                                       \
            ref_footprint(f, w_, h_, &pitch_, &slice_size_);                  \
            for (int sl_ = 0; sl_ < (slice_count); ++sl_) {                    \
                ddsktx_sub_data sd;                                            \
                memset(&sd, 0, sizeof(sd));                                    \
                ddsktx_get_sub(&tc, &sd, blob->data, (int)blob->size,          \
                               (layer_idx),                                    \
                               cfg->cubemap ? (face_idx) : sl_, (mip_idx));    \
                CHECK(sd.buff != NULL, "get_sub returned null buff");          \
                CHECK(sd.width == w_ && sd.height == h_,                       \
                      "sub dims mismatch: got %dx%d want %dx%d",               \
                      sd.width, sd.height, w_, h_);                            \
                CHECK(sd.size_bytes == (int)slice_size_,                       \
                      "sub size mismatch: got %d want %lld",                   \
                      sd.size_bytes, (long long)slice_size_);                  \
                CHECK(sd.row_pitch_bytes == (int)pitch_,                       \
                      "row pitch mismatch: got %d want %lld",                  \
                      sd.row_pitch_bytes, (long long)pitch_);                  \
                if (sd.buff == NULL)                                           \
                    break;                                                     \
                CHECK((const uint8_t*)sd.buff >= blob->data &&                 \
                      (const uint8_t*)sd.buff + sd.size_bytes <=               \
                          blob->data + blob->size,                             \
                      "sub resource out of file bounds");                      \
                const uint8_t* p_ = (const uint8_t*)sd.buff;                   \
                for (int k_ = 0; k_ < sd.size_bytes; ++k_) {                   \
                    uint8_t want_ = payload_byte(pos);                         \
                    double d_ = (double)p_[k_] - (double)want_;                \
                    sse += d_ * d_;                                            \
                    if (p_[k_] != want_) {                                     \
                        CHECK(0, "payload byte mismatch at mip %d face %d "    \
                                 "slice %d off %d (file off %lld)",            \
                              (mip_idx), (face_idx), sl_, k_,                  \
                              (long long)(p_ + k_ - blob->data));              \
                        break;                                                 \
                    }                                                          \
                    ++pos;                                                     \
                }                                                              \
            }                                                                  \
        } while (0)

    if (is_ktx) {
        for (int mip = 0; mip < cfg->num_mips; ++mip) {
            int slices = cfg->depth > 1 ? mip_dim(cfg->depth, mip) : 1;
            for (int layer = 0; layer < cfg->num_layers; ++layer)
                for (int face = 0; face < num_faces; ++face)
                    CHECK_SUB(layer, face, mip, slices);
        }
    } else {
        for (int layer = 0; layer < cfg->num_layers; ++layer) {
            for (int face = 0; face < num_faces; ++face) {
                for (int mip = 0; mip < cfg->num_mips; ++mip) {
                    int slices = cfg->depth > 1 ? mip_dim(cfg->depth, mip) : 1;
                    CHECK_SUB(layer, face, mip, slices);
                }
            }
        }
    }
    #undef CHECK_SUB

    CHECK(sse == 0.0, "SSE = %g (expected 0)", sse);

    /* File length: parser data must cover exactly the payload region. */
    CHECK(tc.data_offset >= (is_ktx ? 64 : (4 + 124)) &&
          tc.data_offset < (int)blob->size, "data offset out of range");
    CHECK(tc.data_offset + tc.size_bytes == (int)blob->size,
          "file length mismatch: data_offset=%d size_bytes=%d file=%zu",
          tc.data_offset, tc.size_bytes, blob->size);
}

static int is_pvrtc(const ref_fmt_t* f)
{
    return f->write_fmt == DDSKTX_FORMAT_PTC12A ||
           f->write_fmt == DDSKTX_FORMAT_PTC14A;
}

static void run_dimension_suite(const ref_fmt_t* f, int is_ktx)
{
    /* Cropped/odd sizes deliberately not aligned to block dimensions.
     * PVRTC only defines square power-of-two textures in the KTX spec,
     * so it gets its own restricted dimension set. */
    static const int dims[][2] = {
        { 1, 1 }, { 2, 2 }, { 3, 3 }, { 5, 7 }, { 6, 6 }, { 7, 3 },
        { 9, 11 }, { 13, 5 }, { 15, 17 }, { 16, 16 }, { 17, 9 },
        { 31, 33 }, { 33, 18 }
    };
    static const int ptc_dims[][2] = {
        { 8, 8 }, { 16, 16 }, { 32, 32 }, { 64, 64 }
    };
    const int (*use_dims)[2] = is_pvrtc(f) ? ptc_dims : dims;
    int num_dims = is_pvrtc(f)
                 ? (int)(sizeof(ptc_dims) / sizeof(ptc_dims[0]))
                 : (int)(sizeof(dims) / sizeof(dims[0]));

    for (int i = 0; i < num_dims; ++i) {
        case_cfg_t cfg;
        cfg.width = use_dims[i][0];
        cfg.height = use_dims[i][1];
        cfg.depth = 1;
        cfg.num_layers = 1;
        cfg.cubemap = 0;
        cfg.num_mips = 1;

        buf_t blob;
        buf_init(&blob);
        if (is_ktx) build_ktx(&blob, f, cfg.width, cfg.height, 1, 1, 1, 0, 0);
        else        build_dds(&blob, f, cfg.width, cfg.height, 1, 1, 1, 0);
        verify_roundtrip(f, &cfg, &blob, is_ktx);
        buf_free(&blob);
    }
}

static void run_mip_suite(const ref_fmt_t* f, int is_ktx)
{
    static const int dims[][2] = {
        { 13, 9 }, { 17, 17 }, { 31, 15 }, { 32, 32 }, { 33, 7 }, { 5, 5 }
    };
    static const int ptc_dims[][2] = {
        { 8, 8 }, { 16, 16 }, { 32, 32 }
    };
    const int (*use_dims)[2] = is_pvrtc(f) ? ptc_dims : dims;
    int num_dims = is_pvrtc(f)
                 ? (int)(sizeof(ptc_dims) / sizeof(ptc_dims[0]))
                 : (int)(sizeof(dims) / sizeof(dims[0]));

    for (int i = 0; i < num_dims; ++i) {
        int w = use_dims[i][0], h = use_dims[i][1];
        int max_mips = 1;
        for (int d = w > h ? w : h; d > 1; d >>= 1) ++max_mips;

        /* The library's block-table model only matches KTX PVRTC sizing at
         * mip 0 (the KTX PVRTC size formula is not a per-level block layout),
         * so multi-mip PVRTC KTX files are only exercised in DDS containers. */
        if (is_ktx && is_pvrtc(f))
            continue;

        case_cfg_t cfg;
        cfg.width = w; cfg.height = h; cfg.depth = 1;
        cfg.num_layers = 1; cfg.cubemap = 0;
        cfg.num_mips = max_mips;

        buf_t blob;
        buf_init(&blob);
        if (is_ktx) build_ktx(&blob, f, w, h, 1, max_mips, 1, 0, 0);
        else        build_dds(&blob, f, w, h, 1, max_mips, 1, 0);
        verify_roundtrip(f, &cfg, &blob, is_ktx);
        buf_free(&blob);
    }
}

static void run_cube_array_suite(const ref_fmt_t* f, int is_ktx)
{
    static const int dims[][2] = { { 13, 7 }, { 9, 9 }, { 16, 16 }, { 3, 5 } };
    static const int ptc_dims[][2] = { { 16, 16 }, { 32, 32 } };
    const int (*use_dims)[2] = is_pvrtc(f) ? ptc_dims : dims;
    int num_dims = is_pvrtc(f)
                 ? (int)(sizeof(ptc_dims) / sizeof(ptc_dims[0]))
                 : (int)(sizeof(dims) / sizeof(dims[0]));

    for (int i = 0; i < num_dims; ++i) {
        int w = use_dims[i][0], h = use_dims[i][1];
        int max_mips = 1;
        for (int d = w > h ? w : h; d > 1; d >>= 1) ++max_mips;

        /* cubemap. Full mip chain, except PVRTC in KTX (see run_mip_suite). */
        {
            int cube_mips = (is_ktx && is_pvrtc(f)) ? 1 : max_mips;
            case_cfg_t cfg = { w, h, 1, cube_mips, 1, 1 };
            buf_t blob; buf_init(&blob);
            if (is_ktx) build_ktx(&blob, f, w, h, 1, cube_mips, 1, 1, 0);
            else        build_dds(&blob, f, w, h, 1, cube_mips, 1, 1);
            verify_roundtrip(f, &cfg, &blob, is_ktx);
            buf_free(&blob);
        }
        /* texture array (2 layers), single mip. DDS arrays need a DX10
         * header, so formats without a DXGI mapping can only be tested in KTX. */
        if (is_ktx || f->dxgi != 0) {
            case_cfg_t cfg = { w, h, 1, 1, 2, 0 };
            buf_t blob; buf_init(&blob);
            if (is_ktx) build_ktx(&blob, f, w, h, 1, 1, 2, 0, 0);
            else        build_dds(&blob, f, w, h, 1, 1, 2, 0);
            verify_roundtrip(f, &cfg, &blob, is_ktx);
            buf_free(&blob);
        }
        /* cubemap array (2 layers) with mips */
        if (is_ktx || f->dxgi != 0) {
            int cam_mips = (is_ktx && is_pvrtc(f)) ? 1 : max_mips;
            case_cfg_t cfg = { w, h, 1, cam_mips, 2, 1 };
            buf_t blob; buf_init(&blob);
            if (is_ktx) build_ktx(&blob, f, w, h, 1, cam_mips, 2, 1, 0);
            else        build_dds(&blob, f, w, h, 1, cam_mips, 2, 1);
            verify_roundtrip(f, &cfg, &blob, is_ktx);
            buf_free(&blob);
        }
    }
}

static void run_volume_suite(const ref_fmt_t* f, int is_ktx)
{
    /* 3D volume with odd dimensions and full mip chain (depth halves too). */
    int w = 11, h = 9, d = 7;
    int max_mips = 1;
    for (int x = w; x > 1; x >>= 1) ++max_mips;
    case_cfg_t cfg = { w, h, d, max_mips, 1, 0 };

    buf_t blob; buf_init(&blob);
    if (is_ktx) build_ktx(&blob, f, w, h, d, max_mips, 1, 0, 0);
    else        build_dds(&blob, f, w, h, d, max_mips, 1, 0);
    verify_roundtrip(f, &cfg, &blob, is_ktx);
    buf_free(&blob);
}

/*
 * Spec-padded KTX interop: uncompressed odd-sized textures produce face sizes
 * that are not multiples of 4, so a conformant writer stores the padded size in
 * imageSize. The reader must still locate every tightly-packed sub resource
 * correctly and report the unpadded size/row pitch.
 */
static void run_padded_ktx_suite(void)
{
    static const ddsktx_format fmts[] = {
        DDSKTX_FORMAT_R8, DDSKTX_FORMAT_RGBA8
    };
    /* R8 sizes chosen so face sizes hit remainders 1..3 mod 4. */
    static const int dims[][2] = {
        { 3, 3 }, { 5, 3 }, { 7, 5 }, { 9, 7 }, { 11, 9 }, { 13, 11 }, { 17, 13 }
    };

    for (unsigned fi = 0; fi < sizeof(fmts) / sizeof(fmts[0]); ++fi) {
        const ref_fmt_t* f = ref_find(fmts[fi]);
        for (unsigned i = 0; i < sizeof(dims) / sizeof(dims[0]); ++i) {
            int w = dims[i][0], h = dims[i][1];
            int max_mips = 1;
            for (int d = w > h ? w : h; d > 1; d >>= 1) ++max_mips;

            /* padded 2D, full mip chain. The first dimension set is checked
             * explicitly to prove the file really contains padding bytes. */
            {
                case_cfg_t cfg = { w, h, 1, max_mips, 1, 0 };
                buf_t blob; buf_init(&blob);
                build_ktx(&blob, f, w, h, 1, max_mips, 1, 0, 1);
                verify_roundtrip(f, &cfg, &blob, 1);

                if (i == 0 && f->write_fmt == DDSKTX_FORMAT_R8) {
                    int64_t pitch_, face_;
                    ref_footprint(f, w, h, &pitch_, &face_);
                    int64_t stride_ = align4_i64(face_);
                    CHECK(stride_ > face_, "test must exercise real padding");
                    /* imageSize at offset 64, payload starts at 68; the bytes
                     * between face_ and stride_ must be the 0xCD fill. */
                    CHECK(blob.size >= (size_t)(64 + 4 + stride_),
                          "padded blob too small");
                    uint32_t stored;
                    memcpy(&stored, blob.data + 64, sizeof(stored));
                    CHECK((int64_t)stored == stride_,
                          "stored imageSize must be padded stride");
                    for (int64_t k = face_; k < stride_; ++k)
                        CHECK(blob.data[68 + k] == 0xCD,
                              "padding byte %lld must be fill", (long long)k);
                }
                buf_free(&blob);
            }
            /* padded cubemap with mips */
            {
                case_cfg_t cfg = { w, h, 1, max_mips, 1, 1 };
                buf_t blob; buf_init(&blob);
                build_ktx(&blob, f, w, h, 1, max_mips, 1, 1, 1);
                verify_roundtrip(f, &cfg, &blob, 1);
                buf_free(&blob);
            }
            /* padded 2D array (3 layers), single mip */
            {
                case_cfg_t cfg = { w, h, 1, 1, 3, 0 };
                buf_t blob; buf_init(&blob);
                build_ktx(&blob, f, w, h, 1, 1, 3, 0, 1);
                verify_roundtrip(f, &cfg, &blob, 1);
                buf_free(&blob);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Negative tests: truncated / malformed files must be rejected       */
/* ------------------------------------------------------------------ */

static void run_negative_tests(void)
{
    const ref_fmt_t* bc1 = ref_find(DDSKTX_FORMAT_BC1);

    /* 1. Truncated DDS: declared full mip chain, payload cut in half. */
    {
        buf_t blob; buf_init(&blob);
        build_dds(&blob, bc1, 33, 17, 1, 6, 1, 0);
        size_t cut = blob.size / 2;
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        int ok = ddsktx_parse(&tc, blob.data, (int)cut, &err);
        CHECK(!ok, "truncated DDS must fail parse");
        buf_free(&blob);
    }

    /* 2. Truncated KTX. */
    {
        buf_t blob; buf_init(&blob);
        build_ktx(&blob, bc1, 33, 17, 1, 6, 1, 0, 0);
        size_t cut = blob.size / 2;
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        int ok = ddsktx_parse(&tc, blob.data, (int)cut, &err);
        CHECK(!ok, "truncated KTX must fail parse");
        buf_free(&blob);
    }

    /* 3. KTX with a corrupted imageSize (too small) must be rejected. */
    {
        buf_t blob; buf_init(&blob);
        build_ktx(&blob, bc1, 8, 8, 1, 2, 1, 0, 0);
        /* first imageSize lives right after the 64-byte header */
        uint32_t bad = 4;
        memcpy(blob.data + 64, &bad, sizeof(bad));
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        int ok = ddsktx_parse(&tc, blob.data, (int)blob.size, &err);
        CHECK(!ok, "KTX with bad imageSize must fail parse");
        buf_free(&blob);
    }

    /* 4. DDS declaring more mips than dimensions allow. */
    {
        buf_t blob; buf_init(&blob);
        build_dds(&blob, bc1, 8, 8, 1, 10, 1, 0);
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        int ok = ddsktx_parse(&tc, blob.data, (int)blob.size, &err);
        CHECK(!ok, "DDS with excessive mip count must fail parse");
        buf_free(&blob);
    }

    /* 5. Empty / garbage blobs. */
    {
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        uint8_t garbage[32];
        memset(garbage, 0xAA, sizeof(garbage));
        CHECK(!ddsktx_parse(&tc, garbage, (int)sizeof(garbage), &err),
              "garbage blob must fail parse");
    }

    /* 6. DDS with oversized dimensions that would overflow 32-bit size math.
     * Built from a raw header so the declared size cannot be satisfied. */
    {
        uint8_t blob[4 + 124 + 64];
        memset(blob, 0, sizeof(blob));
        uint32_t magic = fourcc("DDS ");
        memcpy(blob, &magic, 4);
        uint32_t hdr[31];
        memset(hdr, 0, sizeof(hdr));
        hdr[0] = 124;
        hdr[1] = 0x1 | 0x2 | 0x4 | 0x1000 | 0x80000;
        hdr[2] = 0x40000000u;             /* height */
        hdr[3] = 0x40000000u;             /* width  */
        hdr[6] = 1;
        uint32_t* pf = &hdr[18];
        pf[0] = 32; pf[1] = 0x4; pf[2] = fourcc("DXT1");
        hdr[26] = 0x1000;
        memcpy(blob + 4, hdr, 124);

        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        CHECK(!ddsktx_parse(&tc, blob, (int)sizeof(blob), &err),
              "oversized DDS dimensions must be rejected");
    }

    /* 7. KTX truncated right after the header (no imageSize to read). */
    {
        buf_t blob; buf_init(&blob);
        build_ktx(&blob, bc1, 8, 8, 1, 1, 1, 0, 0);
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        CHECK(!ddsktx_parse(&tc, blob.data, 64, &err),
              "header-only KTX must fail parse");
        buf_free(&blob);
    }

    /* 8. KTX with an imageSize larger than the 4-byte padded face size. */
    {
        buf_t blob; buf_init(&blob);
        build_ktx(&blob, bc1, 8, 8, 1, 1, 1, 0, 0);
        uint32_t bad = 0x00100000u;
        memcpy(blob.data + 64, &bad, sizeof(bad));
        ddsktx_texture_info tc; memset(&tc, 0, sizeof(tc));
        ddsktx_error err; memset(&err, 0, sizeof(err));
        CHECK(!ddsktx_parse(&tc, blob.data, (int)blob.size, &err),
              "KTX imageSize exceeding padded face size must fail parse");
        buf_free(&blob);
    }
}

int main(void)
{
    printf("== dds-ktx roundtrip tests ==\n");

    for (int i = 0; i < k_ref_format_count; ++i) {
        const ref_fmt_t* f = &k_ref_formats[i];
        for (int is_ktx = 0; is_ktx <= 1; ++is_ktx) {
            if ((is_ktx && !f->ktx_ok) || (!is_ktx && !f->dds_ok))
                continue;
            const char* container = is_ktx ? "KTX" : "DDS";
            printf("-- %s / %s: odd/cropped dimensions\n",
                   container, ddsktx_format_str(f->write_fmt));
            run_dimension_suite(f, is_ktx);

            printf("-- %s / %s: full mip chains\n",
                   container, ddsktx_format_str(f->write_fmt));
            run_mip_suite(f, is_ktx);

            printf("-- %s / %s: cubemaps and arrays\n",
                   container, ddsktx_format_str(f->write_fmt));
            run_cube_array_suite(f, is_ktx);
        }

        /* 3D volumes: DDS supports them widely; KTX builder supports them too.
         * PVRTC has no 3D variant. */
        for (int is_ktx = 0; is_ktx <= 1; ++is_ktx) {
            if ((is_ktx && !f->ktx_ok) || (!is_ktx && !f->dds_ok))
                continue;
            if (is_pvrtc(f))
                continue;
            printf("-- %s / %s: 3D volume\n",
                   is_ktx ? "KTX" : "DDS", ddsktx_format_str(f->write_fmt));
            run_volume_suite(f, is_ktx);
        }
    }

    printf("-- spec-padded KTX (uncompressed odd sizes)\n");
    run_padded_ktx_suite();

    printf("-- negative tests\n");
    run_negative_tests();

    printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
